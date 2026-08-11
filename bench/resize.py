"""Derive fixed-height downscaled copies of each source image (the resolution ladder)."""
import os

from PIL import Image


def rungs_for(native_h, rungs):
    """Rungs applicable to an image: no upscaling; tiny images get their native height."""
    fit = [r for r in rungs if r <= native_h]
    return fit or [native_h]


def derive(src_path, image_id, rungs_dir, rungs, quality=85):
    im = None
    outs = []
    for rung in rungs_for(_native_height(src_path), rungs):
        out = os.path.join(rungs_dir, str(rung), image_id)
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if im is None:
                im = Image.open(src_path).convert("RGB")
            w = round(im.width * rung / im.height)
            im.resize((w, rung), Image.LANCZOS).save(out, "JPEG", quality=quality)
        outs.append((rung, out))
    return outs


def _native_height(src_path):
    with Image.open(src_path) as im:
        return im.height
