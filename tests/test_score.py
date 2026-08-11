import bench.score as sc
from bench.config import ModelCfg


def test_iou_basic():
    assert sc.iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert sc.iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    assert abs(sc.iou([0, 0, 100, 100], [50, 0, 150, 100]) - 1 / 3) < 1e-9


def test_greedy_match_same_brand_conf_order():
    truths = [{"brand": "adidas", "box": [0, 0, 100, 100]},
              {"brand": "delay", "box": [500, 500, 600, 600]}]
    dets = [{"brand": "adidas", "box": [510, 510, 590, 590], "conf": 3},  # wrong brand overlap
            {"brand": "adidas", "box": [10, 10, 90, 90], "conf": 2},
            {"brand": "adidas", "box": [5, 5, 95, 95], "conf": 1}]       # duplicate, unmatched
    matches = sc.greedy_match(dets, truths)
    assert matches == [(1, 0, sc.iou([10, 10, 90, 90], [0, 0, 100, 100]))]


def _mk_rows(model, image, rung, dets, parse_ok=True):
    return {"image": image, "rung": rung, "model": model,
            "detections": dets if parse_ok else None, "parse_ok": parse_ok,
            "retried": False, "latency_s": 2.0, "input_tokens": 1000,
            "output_tokens": 100, "error": None, "ts": "t"}


def test_score_all_end_to_end():
    labels = {
        "a.jpg": [{"brand": "adidas", "box": [0, 0, 100, 100], "size": "large",
                   "placement": "foreground", "location": "chest"}],
        "b.jpg": [],  # empty frame, FP control
    }
    det_good = {"brand": "adidas", "box": [10, 10, 90, 90], "size": "large",
                "placement": "foreground", "location": "chest", "conf": 3}
    det_fp = {"brand": "delay", "box": [0, 0, 50, 50], "size": "small",
              "placement": "background", "location": "board", "conf": 1}
    raw = {"m1": [
        _mk_rows("m1", "a.jpg", 480, [det_good]),
        _mk_rows("m1", "b.jpg", 480, [det_fp]),
        _mk_rows("m1", "a.jpg", 240, []),
        _mk_rows("m1", "b.jpg", 240, [], parse_ok=False),
    ]}
    models = [ModelCfg("m1", "openai", "m1", price_in=1.0, price_out=10.0)]
    s = sc.score_all(raw, labels, models, [480, 240])
    r480 = s["models"]["m1"]["rungs"]["480"]
    assert r480["presence"]["adidas"]["f1"] == 1.0
    assert r480["presence"]["delay"]["fp"] == 1          # hallucinated on empty frame
    assert r480["boxes"]["hit03"] == 1.0
    assert r480["attrs"]["size_acc"] == 1.0
    assert r480["ops"]["parse_fail_rate"] == 0.0
    assert abs(r480["ops"]["cost_per_frame"] - (1000 * 1.0 + 100 * 10.0) / 1e6) < 1e-9
    r240 = s["models"]["m1"]["rungs"]["240"]
    assert r240["presence"]["adidas"]["f1"] == 0.0       # missed at 240p
    assert r240["ops"]["parse_fail_rate"] == 0.5
    # retention vs highest rung (480)
    assert s["models"]["m1"]["retention"]["presence_f1"]["240"] == 0.0


def test_retention_guards_zero_division():
    labels = {"a.jpg": []}
    raw = {"m1": [_mk_rows("m1", "a.jpg", 480, []), _mk_rows("m1", "a.jpg", 240, [])]}
    s = sc.score_all(raw, labels, [ModelCfg("m1", "openai", "m1")], [480, 240])
    assert s["models"]["m1"]["retention"]["presence_f1"]["240"] is None
