import json
import os
from PIL import Image
import bench.report as rp


def test_fmt_handles_none():
    assert rp._fmt(None) == "-"
    assert rp._fmt(0.5) == 0.5
    assert rp._fmt("text") == "text"
    assert rp._fmt(0) == 0


def test_disagreements_classifies():
    truth = [{"brand": "adidas", "box": [0, 0, 100, 100]}]
    dets = [{"brand": "adidas", "box": [500, 500, 600, 600], "conf": 3}]
    ds = rp.disagreements(dets, truth)
    kinds = sorted(d["kind"] for d in ds)
    assert kinds == ["model_extra", "truth_missed"]


def test_disagreements_empty_when_matched():
    truth = [{"brand": "adidas", "box": [0, 0, 100, 100]}]
    dets = [{"brand": "adidas", "box": [5, 5, 95, 95], "conf": 3}]
    assert rp.disagreements(dets, truth) == []


def test_render_overlay_writes_jpeg(tmp_path):
    src = tmp_path / "x.jpg"
    Image.new("RGB", (200, 100), "gray").save(src)
    out = tmp_path / "overlay.jpg"
    rp.render_overlay(str(src), [[0, 0, 500, 500]], [[500, 500, 1000, 1000]], str(out))
    assert out.exists() and Image.open(out).size == (200, 100)


def test_apply_reviews_edits_labels(tmp_path):
    labels_dir = tmp_path / "data" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "done": True,
         "boxes": [{"brand": "delay", "box": [1, 1, 9, 9], "size": "small",
                    "placement": "foreground", "location": "chest"}]}))
    reviews = [
        {"entry": {"kind": "truth_missed", "image": "a.jpg", "brand": "delay",
                   "box": [1, 1, 9, 9]}, "verdict": "truth_wrong"},
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "adidas",
                   "box": [100, 100, 200, 200]}, "verdict": "model_right"},
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "dkh",
                   "box": [5, 5, 6, 6]}, "verdict": "both_wrong"},
    ]
    (tmp_path / "data" / "reviews.json").write_text(json.dumps(reviews))
    rp.apply_reviews(str(tmp_path))
    lab = json.load(open(labels_dir / "a.jpg.json"))
    brands = [b["brand"] for b in lab["boxes"]]
    assert brands == ["adidas"]                     # delay removed, adidas added
    assert lab["boxes"][0]["from_review"] is True
    applied_path = tmp_path / "data" / "reviews.applied.json"
    assert applied_path.exists()
    applied = json.load(open(applied_path))
    assert len(applied) == 3                        # all 3 entries archived
    assert applied[0]["applied"] is True            # truth_wrong applied
    assert applied[1]["applied"] is True            # model_right applied
    assert applied[2]["applied"] is False           # both_wrong not applied


def test_apply_reviews_idempotent(tmp_path):
    labels_dir = tmp_path / "data" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "done": True,
         "boxes": [{"brand": "delay", "box": [1, 1, 9, 9], "size": "small",
                    "placement": "foreground", "location": "chest"}]}))
    reviews = [
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "adidas",
                   "box": [100, 100, 200, 200]}, "verdict": "model_right"},
    ]
    (tmp_path / "data" / "reviews.json").write_text(json.dumps(reviews))
    rp.apply_reviews(str(tmp_path))
    lab1 = json.load(open(labels_dir / "a.jpg.json"))
    assert len(lab1["boxes"]) == 2                   # delay + adidas
    assert not os.path.exists(tmp_path / "data" / "reviews.json")
    applied_path = tmp_path / "data" / "reviews.applied.json"
    assert applied_path.exists()
    rp.apply_reviews(str(tmp_path))  # run again
    lab2 = json.load(open(labels_dir / "a.jpg.json"))
    assert lab2["boxes"] == lab1["boxes"]            # no change on second run


def test_apply_reviews_deduplicates_boxes(tmp_path):
    labels_dir = tmp_path / "data" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "done": True,
         "boxes": [{"brand": "adidas", "box": [100, 100, 200, 200], "size": "large",
                    "placement": "background", "location": "sleeve"}]}))
    reviews = [
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "adidas",
                   "box": [100, 100, 200, 200]}, "verdict": "model_right"},
    ]
    (tmp_path / "data" / "reviews.json").write_text(json.dumps(reviews))
    rp.apply_reviews(str(tmp_path))
    lab = json.load(open(labels_dir / "a.jpg.json"))
    assert len(lab["boxes"]) == 1                   # not duplicated
    assert lab["boxes"][0]["brand"] == "adidas"
    assert lab["boxes"][0]["box"] == [100, 100, 200, 200]


def test_render_overlay_tolerates_inverted_boxes(tmp_path):
    from PIL import Image
    src = tmp_path / "x.jpg"
    Image.new("RGB", (200, 100), "gray").save(src)
    out = tmp_path / "overlay.jpg"
    rp.render_overlay(str(src), [[500, 600, 400, 300]], [], str(out))
    assert out.exists()


def test_render_html_includes_charts():
    scores = {"generated": "t", "n_images": 2, "models": {
        "m1": {"rungs": {"480": {
            "presence": {"adidas": {"p": 1.0, "r": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
                         "_macro_f1": 1.0},
            "boxes": {"hit03": 0.8, "hit05": 0.5, "mean_iou": 0.7, "n_truth": 4, "n_det": 4},
            "attrs": {"size_acc": 1.0, "placement_acc": 1.0, "n_matched": 3},
            "ops": {"lat_p50": 2.0, "lat_p95": 3.0, "cost_per_frame": 0.004,
                    "parse_fail_rate": 0.0, "n_frames": 2}},
            "240": {
            "presence": {"adidas": {"p": 0.5, "r": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1},
                         "_macro_f1": 0.5},
            "boxes": {"hit03": None, "hit05": None, "mean_iou": None, "n_truth": 0, "n_det": 1},
            "attrs": {"size_acc": None, "placement_acc": None, "n_matched": 0},
            "ops": {"lat_p50": 1.0, "lat_p95": 2.0, "cost_per_frame": 0.002,
                    "parse_fail_rate": 0.5, "n_frames": 2}}},
            "retention": {"presence_f1": {"240": 0.5}, "hit03": {"240": None}}}}}
    out = rp._render_html(scores, [])
    assert "<svg" in out and "Cost vs quality" in out
    assert "Resolution robustness" in out and "Per-brand presence F1" in out
    assert "data-tt=" in out
