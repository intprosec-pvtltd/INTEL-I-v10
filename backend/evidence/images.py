from __future__ import annotations

from io import BytesIO
import os
from typing import Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from .integrity import sha256_bytes


_MAX_IMAGE_BYTES = max(1_048_576, int(os.getenv("EVIDENCE_REPORT_MAX_SNAPSHOT_BYTES", str(12 * 1024 * 1024))))
_MAX_PIXELS = max(1_000_000, int(os.getenv("EVIDENCE_REPORT_MAX_IMAGE_PIXELS", "20000000")))
_MAX_SIDE = max(512, int(os.getenv("EVIDENCE_REPORT_MAX_IMAGE_SIDE", "3000")))


class EvidenceImageError(ValueError):
    pass


def prepare_snapshot_for_report(data: bytes | None, declared_type: str | None = None) -> Tuple[bytes | None, str | None, str | None]:
    if not data:
        return None, None, None
    if len(data) > _MAX_IMAGE_BYTES:
        raise EvidenceImageError("Evidence snapshot exceeds the configured report size limit")

    digest = sha256_bytes(data)
    old_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            if image.width <= 0 or image.height <= 0 or (image.width * image.height) > _MAX_PIXELS:
                raise EvidenceImageError("Evidence snapshot dimensions exceed the configured report limit")
            image.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "L"}:
                base = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    base.paste(image.convert("RGBA"), mask=image.getchannel("A"))
                else:
                    base.paste(image.convert("RGB"))
                image = base
            elif image.mode == "L":
                image = image.convert("RGB")
            else:
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True, progressive=False)
            return output.getvalue(), "image/jpeg", digest
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise EvidenceImageError("Evidence snapshot is not a valid supported image") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = old_max
