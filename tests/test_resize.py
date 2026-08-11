import os
from PIL import Image
from bench.resize import rungs_for, derive

RUNGS = [1080, 720, 480, 240, 144]


def test_rungs_for_skips_upscales():
    assert rungs_for(1706, RUNGS) == [1080, 720, 480, 240, 144]
    assert rungs_for(1080, RUNGS) == [1080, 720, 480, 240, 144]
    assert rungs_for(500, RUNGS) == [480, 240, 144]


def test_rungs_for_tiny_image_uses_native():
    assert rungs_for(100, RUNGS) == [100]


def test_derive_writes_correct_sizes(tmp_path):
    src = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "red").save(src, "JPEG")
    outs = derive(str(src), "frame.jpg", str(tmp_path / "rungs"), RUNGS)
    assert [r for r, _ in outs] == RUNGS
    for rung, path in outs:
        im = Image.open(path)
        assert im.height == rung
        assert im.width == round(1920 * rung / 1080)


def test_derive_is_idempotent(tmp_path):
    src = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 480), "blue").save(src, "JPEG")
    outs1 = derive(str(src), "frame.jpg", str(tmp_path / "rungs"), RUNGS)
    mtimes = [os.path.getmtime(p) for _, p in outs1]
    outs2 = derive(str(src), "frame.jpg", str(tmp_path / "rungs"), RUNGS)
    assert outs1 == outs2
    assert [os.path.getmtime(p) for _, p in outs2] == mtimes
