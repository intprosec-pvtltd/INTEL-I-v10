from __future__ import annotations

"""
INTEL-I DarkIR low-light restoration adapter.

Upstream model: cidautai/DarkIR (CVPR 2025), MIT licensed.
See third_party/DarkIR-LICENSE.txt and docs/DARKIR_INTEGRATION.md.
This module vendors the inference architecture needed by INTEL-I so the
production backend does not depend on the upstream repository layout.

Pipeline position:
    low-light frame -> DarkIR -> YOLO -> existing INTEL-I pipeline

The checkpoint is intentionally NOT bundled in source control. Set
DARKIR_MODEL_PATH to a trusted DarkIR_384.pt checkpoint.
"""

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Official DarkIR-m inference architecture.
# The parameter names intentionally match the upstream model so the official
# checkpoint can be loaded without conversion.
# ---------------------------------------------------------------------------


class _SimpleGate:
    def __call__(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class _LayerNormFunction:
    @staticmethod
    def apply(x, weight, bias, eps):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)


class _LayerNorm2d:
    def __init__(self, torch_module, channels: int, eps: float = 1e-6):
        nn = torch_module.nn
        self.module = nn.Module()
        self.module.register_parameter("weight", nn.Parameter(torch_module.ones(channels)))
        self.module.register_parameter("bias", nn.Parameter(torch_module.zeros(channels)))
        self.module.eps = eps

    def __call__(self, x):
        return _LayerNormFunction.apply(
            x,
            self.module.weight,
            self.module.bias,
            self.module.eps,
        )


# The following classes are created dynamically against torch.nn so importing
# INTEL-I remains possible on systems where PyTorch is not installed.
def _build_darkir_classes(torch):
    nn = torch.nn
    F = torch.nn.functional

    class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.eps = eps

        def forward(self, x):
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + self.eps).sqrt()
            return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)

    class SimpleGate(nn.Module):
        def forward(self, x):
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class FreMLP(nn.Module):
        def __init__(self, nc, expand=2):
            super().__init__()
            self.process1 = nn.Sequential(
                nn.Conv2d(nc, expand * nc, 1, 1, 0),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(expand * nc, nc, 1, 1, 0),
            )

        def forward(self, x):
            _, _, H, W = x.shape
            x_freq = torch.fft.rfft2(x, norm="backward")
            mag = torch.abs(x_freq)
            pha = torch.angle(x_freq)
            mag = self.process1(mag)
            real = mag * torch.cos(pha)
            imag = mag * torch.sin(pha)
            x_out = torch.complex(real, imag)
            return torch.fft.irfft2(x_out, s=(H, W), norm="backward")

    class Branch(nn.Module):
        def __init__(self, c, DW_Expand, dilation=1):
            super().__init__()
            self.dw_channel = DW_Expand * c
            self.branch = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.dw_channel,
                    out_channels=self.dw_channel,
                    kernel_size=3,
                    padding=dilation,
                    stride=1,
                    groups=self.dw_channel,
                    bias=True,
                    dilation=dilation,
                )
            )

        def forward(self, input):
            return self.branch(input)

    class DBlock(nn.Module):
        def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=None, extra_depth_wise=False):
            super().__init__()
            if dilations is None:
                dilations = [1]
            self.dw_channel = DW_Expand * c
            self.conv1 = nn.Conv2d(c, self.dw_channel, 1, padding=0, stride=1, bias=True)
            self.extra_conv = (
                nn.Conv2d(
                    self.dw_channel,
                    self.dw_channel,
                    3,
                    padding=1,
                    stride=1,
                    groups=c,
                    bias=True,
                    dilation=1,
                )
                if extra_depth_wise
                else nn.Identity()
            )
            self.branches = nn.ModuleList([
                Branch(self.dw_channel, DW_Expand=1, dilation=d) for d in dilations
            ])
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1, padding=0, stride=1, bias=True),
            )
            self.sg1 = SimpleGate()
            self.sg2 = SimpleGate()
            self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1, padding=0, stride=1, bias=True)
            ffn_channel = FFN_Expand * c
            self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, bias=True)
            self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, bias=True)
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)
            self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        def forward(self, inp):
            y = inp
            x = self.norm1(inp)
            x = self.extra_conv(self.conv1(x))
            z = 0
            for branch in self.branches:
                z = z + branch(x)
            z = self.sg1(z)
            x = self.sca(z) * z
            x = self.conv3(x)
            y = inp + self.beta * x
            x = self.conv4(self.norm2(y))
            x = self.sg2(x)
            x = self.conv5(x)
            return y + x * self.gamma

    class EBlock(nn.Module):
        def __init__(self, c, DW_Expand=2, dilations=None, extra_depth_wise=False):
            super().__init__()
            if dilations is None:
                dilations = [1]
            self.dw_channel = DW_Expand * c
            self.extra_conv = (
                nn.Conv2d(c, c, 3, padding=1, stride=1, groups=c, bias=True, dilation=1)
                if extra_depth_wise
                else nn.Identity()
            )
            self.conv1 = nn.Conv2d(c, self.dw_channel, 1, padding=0, stride=1, bias=True)
            self.branches = nn.ModuleList([
                Branch(c, DW_Expand, dilation=d) for d in dilations
            ])
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1, padding=0, stride=1, bias=True),
            )
            self.sg1 = SimpleGate()
            self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1, padding=0, stride=1, bias=True)
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)
            self.freq = FreMLP(nc=c, expand=2)
            self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        def forward(self, inp):
            y = inp
            x = self.norm1(inp)
            x = self.conv1(self.extra_conv(x))
            z = 0
            for branch in self.branches:
                z = z + branch(x)
            z = self.sg1(z)
            x = self.sca(z) * z
            x = self.conv3(x)
            y = inp + self.beta * x
            x_step2 = self.norm2(y)
            x_freq = self.freq(x_step2)
            x = y * x_freq
            return y + x * self.gamma

    class CustomSequential(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.modules_list = nn.ModuleList(args)

        def forward(self, x):
            for module in self.modules_list:
                x = module(x)
            return x

    class DarkIR(nn.Module):
        def __init__(
            self,
            img_channel=3,
            width=32,
            middle_blk_num_enc=2,
            middle_blk_num_dec=2,
            enc_blk_nums=None,
            dec_blk_nums=None,
            dilations=None,
            extra_depth_wise=True,
        ):
            super().__init__()
            enc_blk_nums = [1, 2, 3] if enc_blk_nums is None else enc_blk_nums
            dec_blk_nums = [3, 1, 1] if dec_blk_nums is None else dec_blk_nums
            dilations = [1, 4, 9] if dilations is None else dilations
            self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, bias=True)
            self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, bias=True)
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.middle_blks_enc = CustomSequential(*[
                EBlock(width * 2 ** len(enc_blk_nums), extra_depth_wise=extra_depth_wise)
                for _ in range(middle_blk_num_enc)
            ])
            self.middle_blks_dec = None
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()

            chan = width
            for num in enc_blk_nums:
                self.encoders.append(CustomSequential(*[
                    EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)
                ]))
                self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
                chan *= 2

            # Rebuild middle blocks after the encoder channel count is known.
            self.middle_blks_enc = CustomSequential(*[
                EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)
            ])
            self.middle_blks_dec = CustomSequential(*[
                DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise)
                for _ in range(middle_blk_num_dec)
            ])

            for num in dec_blk_nums:
                self.ups.append(nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                ))
                chan //= 2
                self.decoders.append(CustomSequential(*[
                    DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise)
                    for _ in range(num)
                ]))

            self.padder_size = 2 ** len(self.encoders)
            self.side_out = nn.Conv2d(
                width * 2 ** len(self.encoders), 3, 3, stride=1, padding=1
            )

        def forward(self, input, side_loss=False, use_adapter=None):
            _, _, H, W = input.shape
            input_padded = self.check_image_size(input)
            x = self.intro(input_padded)
            skips = []
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                skips.append(x)
                x = down(x)
            x_light = self.middle_blks_enc(x)
            if side_loss:
                out_side = self.side_out(x_light)
            x = self.middle_blks_dec(x_light)
            x = x + x_light
            for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
                x = up(x)
                x = x + skip
                x = decoder(x)
            x = self.ending(x)
            x = x + input_padded
            out = x[:, :, :H, :W]
            return (out_side, out) if side_loss else out

        def check_image_size(self, x):
            _, _, h, w = x.size()
            mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
            mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
            return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)

    return DarkIR


class DarkIRRestorer:
    """Lazy, thread-safe DarkIR-m inference service for INTEL-I."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        max_width: int = 1280,
        max_height: int = 720,
        strict: bool = False,
        expected_sha256: str = "",
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = str(device or "auto").strip().lower()
        self.max_width = max(320, int(max_width))
        self.max_height = max(240, int(max_height))
        self.strict = bool(strict)
        self.expected_sha256 = str(expected_sha256 or "").strip().lower()
        self._lock = threading.Lock()
        self._model = None
        self._torch = None
        self._device = "unloaded"
        self._load_error: Optional[str] = None
        self._load_attempted = False
        self._last_ms = 0.0
        self._runs = 0

    def _resolve_device(self):
        torch = self._torch
        if self.requested_device in {"cpu", "none"}:
            return torch.device("cpu")
        if self.requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("DARKIR_DEVICE requests CUDA but torch.cuda.is_available() is false")
            return torch.device(self.requested_device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True

            if not self.model_path.is_file():
                self._load_error = f"DarkIR checkpoint not found: {self.model_path}"
                logger.warning("[INTEL-I][DARKIR] %s", self._load_error)
                if self.strict:
                    raise FileNotFoundError(self._load_error)
                return False

            try:
                import torch
                self._torch = torch
                device = self._resolve_device()
                if self.expected_sha256:
                    actual = _sha256(self.model_path)
                    if actual != self.expected_sha256:
                        raise RuntimeError(
                            f"DarkIR checkpoint SHA256 mismatch: expected={self.expected_sha256} actual={actual}"
                        )

                DarkIR = _build_darkir_classes(torch)
                model = DarkIR(
                    img_channel=3,
                    width=32,
                    middle_blk_num_enc=2,
                    middle_blk_num_dec=2,
                    enc_blk_nums=[1, 2, 3],
                    dec_blk_nums=[3, 1, 1],
                    dilations=[1, 4, 9],
                    extra_depth_wise=True,
                )

                # weights_only=True avoids executing arbitrary Python objects
                # embedded in an untrusted checkpoint.
                checkpoint = torch.load(
                    str(self.model_path),
                    map_location="cpu",
                    weights_only=True,
                )
                state = checkpoint.get("params") if isinstance(checkpoint, dict) else None
                if state is None and isinstance(checkpoint, dict):
                    state = checkpoint.get("state_dict")
                if state is None and isinstance(checkpoint, dict):
                    state = checkpoint
                if not isinstance(state, dict):
                    raise RuntimeError("DarkIR checkpoint does not contain a usable state dictionary")

                # Accept a DDP checkpoint only when the keys clearly use the
                # module. prefix; never silently ignore missing parameters.
                if state and all(str(k).startswith("module.") for k in state.keys()):
                    state = {str(k)[7:]: v for k, v in state.items()}

                expected_keys = set(model.state_dict().keys())
                received_keys = set(state.keys())
                missing = sorted(expected_keys - received_keys)
                unexpected = sorted(received_keys - expected_keys)
                if missing or unexpected:
                    raise RuntimeError(
                        f"DarkIR checkpoint incompatible: missing={missing[:8]} unexpected={unexpected[:8]}"
                    )

                model.load_state_dict(state, strict=True)
                model.eval().to(device)
                self._model = model
                self._device = str(device)
                logger.info(
                    "[INTEL-I][DARKIR] loaded | model=DarkIR-m | device=%s | checkpoint=%s",
                    self._device,
                    self.model_path,
                )
                return True
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("[INTEL-I][DARKIR] load failed")
                if self.strict:
                    raise
                return False

    def _resize_for_model(self, rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = rgb.shape[:2]
        scale = min(1.0, self.max_width / max(1, w), self.max_height / max(1, h))
        if scale >= 0.999:
            return rgb, (w, h)
        nw = max(8, int(round(w * scale / 8.0) * 8))
        nh = max(8, int(round(h * scale / 8.0) * 8))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        return resized, (w, h)

    def enhance(self, frame: np.ndarray) -> Dict[str, Any]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return {"frame": frame, "model_used": False, "reason": "empty_frame"}
        if not self.load():
            return {
                "frame": frame,
                "model_used": False,
                "reason": self._load_error or "darkir_unavailable",
            }

        torch = self._torch
        original = frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model_input, original_size = self._resize_for_model(rgb)
            tensor = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0
            tensor = tensor.to(self._model.device if hasattr(self._model, "device") else self._device)

            started = time.perf_counter()
            with torch.inference_mode():
                output = self._model(tensor, side_loss=False)
            if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            output = torch.clamp(output, 0.0, 1.0)
            out = output[0].permute(1, 2, 0).detach().float().cpu().numpy()
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            original_w, original_h = original_size
            if out.shape[1] != original_w or out.shape[0] != original_h:
                out = cv2.resize(out, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            self._last_ms = elapsed_ms
            self._runs += 1
            return {
                "frame": np.ascontiguousarray(out),
                "model_used": True,
                "model": "DarkIR-m",
                "device": self._device,
                "inference_ms": round(elapsed_ms, 3),
                "resized_for_model": bool(model_input.shape[:2] != rgb.shape[:2]),
            }
        except Exception as exc:
            logger.exception("[INTEL-I][DARKIR] inference failed")
            return {
                "frame": original,
                "model_used": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "checkpoint": str(self.model_path),
            "checkpoint_exists": self.model_path.is_file(),
            "requested_device": self.requested_device,
            "actual_device": self._device,
            "loaded": self._model is not None,
            "load_attempted": self._load_attempted,
            "load_error": self._load_error,
            "runs": self._runs,
            "last_inference_ms": round(self._last_ms, 3),
            "max_width": self.max_width,
            "max_height": self.max_height,
        }
