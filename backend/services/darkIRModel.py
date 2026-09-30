from __future__ import annotations

"""
INTEL-I DarkIR Model
====================

Production-safe DarkIR-m model architecture and checkpoint loader.

Upstream:
    cidautai/DarkIR
    CVPR 2025
    MIT License

INTEL-I pipeline position:

    LOW-LIGHT / POOR QUALITY
                |
                v
             DarkIR
                |
                v
               YOLO
                |
                v
        Existing INTEL-I
        processing pipeline

Important:
    - This module contains the model architecture.
    - It does NOT download model weights.
    - It does NOT execute arbitrary Python from checkpoints.
    - Checkpoint loading uses weights_only=True.
    - State-dict keys are validated strictly.
    - Optional SHA-256 verification is supported.
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class DarkIRConfig:
    """
    Official DarkIR-m architecture configuration.

    These values correspond to the DarkIR-m configuration documented by
    the upstream project.
    """

    img_channel: int = 3
    width: int = 32

    middle_blk_num_enc: int = 2
    middle_blk_num_dec: int = 2

    enc_blk_nums: Tuple[int, ...] = (1, 2, 3)
    dec_blk_nums: Tuple[int, ...] = (3, 1, 1)

    dilations: Tuple[int, ...] = (1, 4, 9)

    extra_depth_wise: bool = True

    @property
    def padder_size(self) -> int:
        return 2 ** len(self.enc_blk_nums)


DEFAULT_DARKIR_CONFIG = DarkIRConfig()


# ============================================================================
# Utility functions
# ============================================================================

def calculate_sha256(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 for a model checkpoint.

    Reads incrementally so the entire checkpoint is never loaded into RAM
    just for hashing.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"DarkIR checkpoint does not exist: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def verify_sha256(
    file_path: str | Path,
    expected_sha256: str,
) -> bool:
    """
    Verify checkpoint SHA-256.

    The expected hash must be a 64-character hexadecimal SHA-256 value.
    """

    expected = str(expected_sha256).strip().lower()

    if len(expected) != 64:
        raise ValueError(
            "Expected DarkIR SHA-256 must contain exactly 64 hexadecimal characters."
        )

    try:
        int(expected, 16)
    except ValueError as exc:
        raise ValueError(
            "Expected DarkIR SHA-256 contains non-hexadecimal characters."
        ) from exc

    actual = calculate_sha256(file_path)

    return actual == expected


# ============================================================================
# DarkIR primitive layers
# ============================================================================

def _require_torch():
    """
    Import PyTorch lazily.

    This keeps INTEL-I importable in environments where PyTorch is not
    installed, provided DarkIR itself is not instantiated.
    """

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F

    except ImportError as exc:
        raise RuntimeError(
            "DarkIR requires PyTorch. Install a compatible PyTorch/CUDA "
            "environment before enabling DarkIR."
        ) from exc


# ============================================================================
# LayerNorm2d
# ============================================================================

def build_layer_norm_2d(torch_module, nn_module):
    """
    Build the LayerNorm2d implementation used by DarkIR.
    """

    torch = torch_module
    nn = nn_module

    class LayerNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            n, c, h, w = x.size()

            mu = x.mean(1, keepdim=True)

            var = (x - mu).pow(2).mean(1, keepdim=True)

            y = (x - mu) / (var + eps).sqrt()

            ctx.eps = eps

            ctx.save_for_backward(
                y,
                var,
                weight,
            )

            y = (
                weight.view(1, c, 1, 1) * y
                + bias.view(1, c, 1, 1)
            )

            return y

        @staticmethod
        def backward(ctx, grad_output):
            eps = ctx.eps

            y, var, weight = ctx.saved_tensors

            g = (
                grad_output
                * weight.view(1, -1, 1, 1)
            )

            mean_g = g.mean(
                dim=1,
                keepdim=True,
            )

            mean_gy = (
                g * y
            ).mean(
                dim=1,
                keepdim=True,
            )

            gx = (
                1.0
                / torch.sqrt(var + eps)
                * (
                    g
                    - y * mean_gy
                    - mean_g
                )
            )

            grad_weight = (
                grad_output * y
            ).sum(
                dim=3
            ).sum(
                dim=2
            ).sum(
                dim=0
            )

            grad_bias = (
                grad_output
            ).sum(
                dim=3
            ).sum(
                dim=2
            ).sum(
                dim=0
            )

            return (
                gx,
                grad_weight,
                grad_bias,
                None,
            )

    class LayerNorm2d(nn.Module):
        def __init__(
            self,
            channels: int,
            eps: float = 1e-6,
        ):
            super().__init__()

            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.ones(channels)
                ),
            )

            self.register_parameter(
                "bias",
                nn.Parameter(
                    torch.zeros(channels)
                ),
            )

            self.eps = eps

        def forward(self, x):
            return LayerNormFunction.apply(
                x,
                self.weight,
                self.bias,
                self.eps,
            )

    return LayerNorm2d


# ============================================================================
# DarkIR architecture
# ============================================================================

def build_darkir_class(
    config: DarkIRConfig = DEFAULT_DARKIR_CONFIG,
):
    """
    Build the official DarkIR architecture dynamically.

    Returning a class instead of importing the upstream repository keeps the
    INTEL-I backend self-contained.
    """

    torch, nn, F = _require_torch()

    LayerNorm2d = build_layer_norm_2d(
        torch,
        nn,
    )

    # ------------------------------------------------------------------------
    # SimpleGate
    # ------------------------------------------------------------------------

    class SimpleGate(nn.Module):
        def forward(self, x):
            x1, x2 = x.chunk(
                2,
                dim=1,
            )

            return x1 * x2

    # ------------------------------------------------------------------------
    # Adapter
    # ------------------------------------------------------------------------

    class Adapter(nn.Module):
        """
        Kept for checkpoint/model compatibility with the upstream architecture.

        Current DarkIR inference does not enable adapters.
        """

        def __init__(
            self,
            c: int,
            ffn_channel: Optional[int] = None,
        ):
            super().__init__()

            if ffn_channel:
                ffn_channel = 2
            else:
                ffn_channel = c

            self.conv1 = nn.Conv2d(
                in_channels=c,
                out_channels=ffn_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )

            self.conv2 = nn.Conv2d(
                in_channels=ffn_channel,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )

            self.depthwise = nn.Conv2d(
                in_channels=c,
                out_channels=ffn_channel,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=c,
                bias=True,
                dilation=1,
            )

        def forward(self, input_tensor):
            x = (
                self.conv1(input_tensor)
                + self.depthwise(input_tensor)
            )

            x = self.conv2(x)

            return x

    # ------------------------------------------------------------------------
    # Frequency MLP
    # ------------------------------------------------------------------------

    class FreMLP(nn.Module):
        def __init__(
            self,
            nc: int,
            expand: int = 2,
        ):
            super().__init__()

            self.process1 = nn.Sequential(
                nn.Conv2d(
                    nc,
                    expand * nc,
                    1,
                    1,
                    0,
                ),
                nn.LeakyReLU(
                    0.1,
                    inplace=True,
                ),
                nn.Conv2d(
                    expand * nc,
                    nc,
                    1,
                    1,
                    0,
                ),
            )

        def forward(self, x):
            _, _, h, w = x.shape

            x_freq = torch.fft.rfft2(
                x,
                norm="backward",
            )

            magnitude = torch.abs(
                x_freq
            )

            phase = torch.angle(
                x_freq
            )

            magnitude = self.process1(
                magnitude
            )

            real = (
                magnitude
                * torch.cos(phase)
            )

            imaginary = (
                magnitude
                * torch.sin(phase)
            )

            complex_output = torch.complex(
                real,
                imaginary,
            )

            output = torch.fft.irfft2(
                complex_output,
                s=(h, w),
                norm="backward",
            )

            return output

    # ------------------------------------------------------------------------
    # Dilated branch
    # ------------------------------------------------------------------------

    class Branch(nn.Module):
        def __init__(
            self,
            c: int,
            dw_expand: int,
            dilation: int = 1,
        ):
            super().__init__()

            dw_channel = dw_expand * c

            self.branch = nn.Sequential(
                nn.Conv2d(
                    in_channels=dw_channel,
                    out_channels=dw_channel,
                    kernel_size=3,
                    padding=dilation,
                    stride=1,
                    groups=dw_channel,
                    bias=True,
                    dilation=dilation,
                )
            )

        def forward(self, input_tensor):
            return self.branch(
                input_tensor
            )

    # ------------------------------------------------------------------------
    # Encoder block
    # ------------------------------------------------------------------------

    class EBlock(nn.Module):
        def __init__(
            self,
            c: int,
            dw_expand: int = 2,
            dilations: Sequence[int] = (1,),
            extra_depth_wise: bool = False,
        ):
            super().__init__()

            self.dw_channel = dw_expand * c

            if extra_depth_wise:
                self.extra_conv = nn.Conv2d(
                    c,
                    c,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=c,
                    bias=True,
                    dilation=1,
                )
            else:
                self.extra_conv = nn.Identity()

            self.conv1 = nn.Conv2d(
                in_channels=c,
                out_channels=self.dw_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            )

            self.branches = nn.ModuleList(
                [
                    Branch(
                        c,
                        dw_expand,
                        dilation=dilation,
                    )
                    for dilation in dilations
                ]
            )

            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    in_channels=self.dw_channel // 2,
                    out_channels=self.dw_channel // 2,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True,
                    dilation=1,
                ),
            )

            self.sg1 = SimpleGate()

            self.conv3 = nn.Conv2d(
                in_channels=self.dw_channel // 2,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            )

            self.norm1 = LayerNorm2d(c)

            self.norm2 = LayerNorm2d(c)

            self.freq = FreMLP(
                nc=c,
                expand=2,
            )

            self.gamma = nn.Parameter(
                torch.zeros(
                    (1, c, 1, 1),
                ),
                requires_grad=True,
            )

            self.beta = nn.Parameter(
                torch.zeros(
                    (1, c, 1, 1),
                ),
                requires_grad=True,
            )

        def forward(self, inp):
            y = inp

            x = self.norm1(
                inp
            )

            x = self.conv1(
                self.extra_conv(x)
            )

            z = None

            for branch in self.branches:
                branch_output = branch(x)

                if z is None:
                    z = branch_output
                else:
                    z = z + branch_output

            z = self.sg1(z)

            x = self.sca(z) * z

            x = self.conv3(x)

            y = (
                inp
                + self.beta * x
            )

            x_step2 = self.norm2(y)

            x_freq = self.freq(
                x_step2
            )

            x = y * x_freq

            return (
                y
                + x * self.gamma
            )

    # ------------------------------------------------------------------------
    # Decoder block
    # ------------------------------------------------------------------------

    class DBlock(nn.Module):
        def __init__(
            self,
            c: int,
            dw_expand: int = 2,
            ffn_expand: int = 2,
            dilations: Sequence[int] = (1,),
            extra_depth_wise: bool = False,
        ):
            super().__init__()

            self.dw_channel = dw_expand * c

            self.conv1 = nn.Conv2d(
                in_channels=c,
                out_channels=self.dw_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            )

            if extra_depth_wise:
                self.extra_conv = nn.Conv2d(
                    self.dw_channel,
                    self.dw_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=c,
                    bias=True,
                    dilation=1,
                )
            else:
                self.extra_conv = nn.Identity()

            self.branches = nn.ModuleList(
                [
                    Branch(
                        self.dw_channel,
                        dw_expand=1,
                        dilation=dilation,
                    )
                    for dilation in dilations
                ]
            )

            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    in_channels=self.dw_channel // 2,
                    out_channels=self.dw_channel // 2,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True,
                    dilation=1,
                ),
            )

            self.sg1 = SimpleGate()

            self.sg2 = SimpleGate()

            self.conv3 = nn.Conv2d(
                in_channels=self.dw_channel // 2,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            )

            ffn_channel = ffn_expand * c

            self.conv4 = nn.Conv2d(
                in_channels=c,
                out_channels=ffn_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )

            self.conv5 = nn.Conv2d(
                in_channels=ffn_channel // 2,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            )

            self.norm1 = LayerNorm2d(c)

            self.norm2 = LayerNorm2d(c)

            self.gamma = nn.Parameter(
                torch.zeros(
                    (1, c, 1, 1),
                ),
                requires_grad=True,
            )

            self.beta = nn.Parameter(
                torch.zeros(
                    (1, c, 1, 1),
                ),
                requires_grad=True,
            )

        def forward(self, inp):
            y = inp

            x = self.norm1(
                inp
            )

            x = self.extra_conv(
                self.conv1(x)
            )

            z = None

            for branch in self.branches:
                branch_output = branch(x)

                if z is None:
                    z = branch_output
                else:
                    z = z + branch_output

            z = self.sg1(z)

            x = self.sca(z) * z

            x = self.conv3(x)

            y = (
                inp
                + self.beta * x
            )

            x = self.conv4(
                self.norm2(y)
            )

            x = self.sg2(x)

            x = self.conv5(x)

            return (
                y
                + x * self.gamma
            )

    # ------------------------------------------------------------------------
    # Custom sequential
    # ------------------------------------------------------------------------

    class CustomSequential(nn.Module):
        """
        Sequential container compatible with the upstream DarkIR structure.
        """

        def __init__(self, *modules):
            super().__init__()

            self.modules_list = nn.ModuleList(
                modules
            )

        def forward(
            self,
            x,
            use_adapter=False,
        ):
            for module in self.modules_list:
                if hasattr(
                    module,
                    "set_use_adapters",
                ):
                    module.set_use_adapters(
                        use_adapter
                    )

                x = module(x)

            return x

    # ------------------------------------------------------------------------
    # DarkIR model
    # ------------------------------------------------------------------------

    class DarkIR(nn.Module):
        """
        DarkIR-m.

        Architecture:
            width=32
            enc=[1,2,3]
            dec=[3,1,1]
            middle encoder=2
            middle decoder=2
            dilations=[1,4,9]
        """

        def __init__(
            self,
            img_channel: int = config.img_channel,
            width: int = config.width,
            middle_blk_num_enc: int = config.middle_blk_num_enc,
            middle_blk_num_dec: int = config.middle_blk_num_dec,
            enc_blk_nums: Sequence[int] = config.enc_blk_nums,
            dec_blk_nums: Sequence[int] = config.dec_blk_nums,
            dilations: Sequence[int] = config.dilations,
            extra_depth_wise: bool = config.extra_depth_wise,
        ):
            super().__init__()

            self.intro = nn.Conv2d(
                in_channels=img_channel,
                out_channels=width,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            )

            self.ending = nn.Conv2d(
                in_channels=width,
                out_channels=img_channel,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            )

            self.encoders = nn.ModuleList()

            self.decoders = nn.ModuleList()

            self.middle_blks_enc = None

            self.middle_blks_dec = None

            self.ups = nn.ModuleList()

            self.downs = nn.ModuleList()

            chan = width

            # ---------------------------------------------------------------
            # Encoder
            # ---------------------------------------------------------------

            for num in enc_blk_nums:

                self.encoders.append(
                    CustomSequential(
                        *[
                            EBlock(
                                chan,
                                extra_depth_wise=extra_depth_wise,
                            )
                            for _ in range(num)
                        ]
                    )
                )

                self.downs.append(
                    nn.Conv2d(
                        chan,
                        2 * chan,
                        kernel_size=2,
                        stride=2,
                    )
                )

                chan = chan * 2

            # ---------------------------------------------------------------
            # Middle encoder
            # ---------------------------------------------------------------

            self.middle_blks_enc = (
                CustomSequential(
                    *[
                        EBlock(
                            chan,
                            extra_depth_wise=extra_depth_wise,
                        )
                        for _ in range(
                            middle_blk_num_enc
                        )
                    ]
                )
            )

            # ---------------------------------------------------------------
            # Middle decoder
            # ---------------------------------------------------------------

            self.middle_blks_dec = (
                CustomSequential(
                    *[
                        DBlock(
                            chan,
                            dilations=dilations,
                            extra_depth_wise=extra_depth_wise,
                        )
                        for _ in range(
                            middle_blk_num_dec
                        )
                    ]
                )
            )

            # ---------------------------------------------------------------
            # Decoder
            # ---------------------------------------------------------------

            for num in dec_blk_nums:

                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(
                            chan,
                            chan * 2,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.PixelShuffle(2),
                    )
                )

                chan = chan // 2

                self.decoders.append(
                    CustomSequential(
                        *[
                            DBlock(
                                chan,
                                dilations=dilations,
                                extra_depth_wise=extra_depth_wise,
                            )
                            for _ in range(num)
                        ]
                    )
                )

            self.padder_size = (
                2 ** len(self.encoders)
            )

            # ---------------------------------------------------------------
            # Side output used by the original training implementation.
            # Not required for normal INTEL-I inference.
            # ---------------------------------------------------------------

            self.side_out = nn.Conv2d(
                in_channels=(
                    width
                    * 2 ** len(self.encoders)
                ),
                out_channels=img_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        def check_image_size(self, x):
            _, _, h, w = x.size()

            mod_pad_h = (
                self.padder_size
                - h % self.padder_size
            ) % self.padder_size

            mod_pad_w = (
                self.padder_size
                - w % self.padder_size
            ) % self.padder_size

            return F.pad(
                x,
                (
                    0,
                    mod_pad_w,
                    0,
                    mod_pad_h,
                ),
                value=0,
            )

        def forward(
            self,
            input_tensor,
            side_loss: bool = False,
            use_adapter=None,
        ):
            _, _, original_h, original_w = (
                input_tensor.shape
            )

            padded_input = (
                self.check_image_size(
                    input_tensor
                )
            )

            x = self.intro(
                padded_input
            )

            skips = []

            # Encoder
            for encoder, down in zip(
                self.encoders,
                self.downs,
            ):
                x = encoder(x)

                skips.append(x)

                x = down(x)

            # Middle encoder
            x_light = (
                self.middle_blks_enc(x)
            )

            if side_loss:
                out_side = self.side_out(
                    x_light
                )

            # Middle decoder
            x = self.middle_blks_dec(
                x_light
            )

            x = (
                x
                + x_light
            )

            # Decoder
            for decoder, up, skip in zip(
                self.decoders,
                self.ups,
                reversed(skips),
            ):
                x = up(x)

                x = (
                    x
                    + skip
                )

                x = decoder(x)

            x = self.ending(x)

            # Residual connection
            x = (
                x
                + padded_input
            )

            # Restore original image dimensions
            output = x[
                :,
                :,
                :original_h,
                :original_w,
            ]

            if side_loss:
                return out_side, output

            return output

    return DarkIR


# ============================================================================
# Model factory
# ============================================================================

def create_darkir_model(
    config: DarkIRConfig = DEFAULT_DARKIR_CONFIG,
):
    """
    Create an uninitialized DarkIR model.

    The caller is responsible for loading the trusted checkpoint.
    """

    DarkIR = build_darkir_class(
        config
    )

    model = DarkIR(
        img_channel=config.img_channel,
        width=config.width,
        middle_blk_num_enc=config.middle_blk_num_enc,
        middle_blk_num_dec=config.middle_blk_num_dec,
        enc_blk_nums=config.enc_blk_nums,
        dec_blk_nums=config.dec_blk_nums,
        dilations=config.dilations,
        extra_depth_wise=config.extra_depth_wise,
    )

    return model


# ============================================================================
# Checkpoint handling
# ============================================================================

def _extract_state_dict(
    checkpoint: Any,
) -> Mapping[str, Any]:
    """
    Extract a model state dictionary from a trusted PyTorch checkpoint.

    Supported formats:

        {
            "params": {...}
        }

    or:

        {
            "state_dict": {...}
        }

    or a direct state dictionary.

    No executable Python objects are accepted.
    """

    if not isinstance(
        checkpoint,
        Mapping,
    ):
        raise RuntimeError(
            "DarkIR checkpoint must deserialize to a mapping."
        )

    if "params" in checkpoint:
        state = checkpoint["params"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    else:
        state = checkpoint

    if not isinstance(
        state,
        Mapping,
    ):
        raise RuntimeError(
            "DarkIR checkpoint does not contain a valid state dictionary."
        )

    return state


def _normalize_state_dict_keys(
    state_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Normalize common DataParallel/DDP prefixes.

    Only known prefixes are removed.
    """

    normalized: Dict[str, Any] = {}

    for key, value in state_dict.items():

        key_string = str(key)

        if key_string.startswith(
            "module."
        ):
            key_string = key_string[
                len("module.") :
            ]

        normalized[key_string] = value

    return normalized


def _validate_state_dict(
    model,
    state_dict: Mapping[str, Any],
) -> None:
    """
    Strictly validate checkpoint/model compatibility.

    No missing or unexpected keys are silently ignored.
    """

    expected = set(
        model.state_dict().keys()
    )

    received = set(
        state_dict.keys()
    )

    missing = sorted(
        expected - received
    )

    unexpected = sorted(
        received - expected
    )

    if missing or unexpected:

        missing_preview = missing[:20]

        unexpected_preview = unexpected[:20]

        raise RuntimeError(
            "DarkIR checkpoint is incompatible with the "
            "INTEL-I DarkIR architecture. "
            f"missing={missing_preview}; "
            f"unexpected={unexpected_preview}"
        )

    # Validate that all checkpoint values are tensors.
    for key, value in state_dict.items():

        if not hasattr(
            value,
            "shape",
        ):
            raise RuntimeError(
                f"DarkIR checkpoint parameter '{key}' "
                "is not tensor-like."
            )


def load_darkir_checkpoint(
    model,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = "",
    device: str = "cpu",
):
    """
    Securely load a DarkIR checkpoint.

    Security properties:
        - Optional SHA-256 verification.
        - weights_only=True.
        - CPU-first checkpoint deserialization.
        - Strict state-dict validation.
        - No arbitrary checkpoint code execution.
        - Model switched to eval mode.
        - No optimizer/training state is loaded.
    """

    torch, _, _ = _require_torch()

    path = Path(
        checkpoint_path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"DarkIR checkpoint not found: {path}"
        )

    # ------------------------------------------------------------------------
    # Optional cryptographic verification
    # ------------------------------------------------------------------------

    if expected_sha256:

        if not verify_sha256(
            path,
            expected_sha256,
        ):
            actual = calculate_sha256(
                path
            )

            raise RuntimeError(
                "DarkIR checkpoint SHA-256 verification failed. "
                f"expected={expected_sha256.lower()} "
                f"actual={actual}"
            )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # weights_only=True prevents the unsafe legacy behavior of loading
    # arbitrary Python objects from an untrusted pickle checkpoint.
    # ------------------------------------------------------------------------

    try:

        checkpoint = torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
        )

    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support secure "
            "weights_only checkpoint loading. "
            "Use a supported PyTorch 2.x version rather than "
            "falling back to unsafe pickle loading."
        ) from exc

    state_dict = _extract_state_dict(
        checkpoint
    )

    state_dict = _normalize_state_dict_keys(
        state_dict
    )

    _validate_state_dict(
        model,
        state_dict,
    )

    # Strict load — never silently skip weights.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    target_device = torch.device(
        device
    )

    model = model.to(
        target_device
    )

    model.eval()

    return model


# ============================================================================
# High-level loader
# ============================================================================

def load_darkir(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    expected_sha256: str = "",
    config: DarkIRConfig = DEFAULT_DARKIR_CONFIG,
):
    """
    Create and securely load DarkIR-m.

    Example:

        model = load_darkir(
            "./models/DarkIR_384.pt",
            device="cuda:0",
            expected_sha256="...",
        )
    """

    model = create_darkir_model(
        config
    )

    model = load_darkir_checkpoint(
        model,
        checkpoint_path,
        expected_sha256=expected_sha256,
        device=device,
    )

    return model


# ============================================================================
# Model information
# ============================================================================

def count_parameters(
    model,
) -> int:
    """
    Return the number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def get_darkir_model_info(
    model,
) -> Dict[str, Any]:
    """
    Return useful runtime/model information.
    """

    return {
        "model": "DarkIR-m",
        "parameters": count_parameters(
            model
        ),
        "training": bool(
            model.training
        ),
        "padder_size": getattr(
            model,
            "padder_size",
            None,
        ),
    }


# ============================================================================
# Self-test
# ============================================================================

def self_test(
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run a small architecture-only validation.

    This does NOT require model weights.

    It verifies:
        - model construction
        - forward pass
        - input/output dimensions
        - parameter count
    """

    torch, _, _ = _require_torch()

    model = create_darkir_model()

    model = model.to(
        torch.device(device)
    )

    model.eval()

    # Non-multiple-of-8 dimensions deliberately test DarkIR padding.
    input_tensor = torch.rand(
        1,
        3,
        257,
        341,
        device=device,
    )

    with torch.inference_mode():

        output = model(
            input_tensor,
            side_loss=False,
        )

    if tuple(
        output.shape
    ) != tuple(
        input_tensor.shape
    ):
        raise RuntimeError(
            "DarkIR self-test failed: "
            f"input_shape={tuple(input_tensor.shape)} "
            f"output_shape={tuple(output.shape)}"
        )

    return {
        "ok": True,
        "model": "DarkIR-m",
        "parameters": count_parameters(
            model
        ),
        "input_shape": tuple(
            input_tensor.shape
        ),
        "output_shape": tuple(
            output.shape
        ),
        "device": str(
            device
        ),
        "padder_size": model.padder_size,
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )

    try:

        result = self_test(
            device="cpu"
        )

        print(
            "DarkIR self-test PASSED"
        )

        for key, value in result.items():
            print(
                f"{key}: {value}"
            )

    except Exception as exc:

        logger.exception(
            "DarkIR self-test FAILED"
        )

        raise