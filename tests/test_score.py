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
    s = sc.score_all(raw, labels, models, [480, 240], brand_universe=["adidas", "delay"])
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


def test_retention_guards_none_numerator():
    """Regression test for FINDING 1: retention should not crash when cur is None but base is float."""
    labels = {
        "a.jpg": [{"brand": "adidas", "box": [0, 0, 100, 100]}],  # has truth
        "b.jpg": [],  # empty truth
    }
    det_match = {"brand": "adidas", "box": [10, 10, 90, 90], "conf": 3}
    raw = {"m1": [
        # 480: has matched detection, hit03 = 1.0
        _mk_rows("m1", "a.jpg", 480, [det_match]),
        _mk_rows("m1", "b.jpg", 480, []),
        # 240: only empty-truth images, hit03 = None
        _mk_rows("m1", "b.jpg", 240, []),
    ]}
    models = [ModelCfg("m1", "openai", "m1")]
    s = sc.score_all(raw, labels, models, [480, 240])
    # 480: n_truth=1, hit03=1.0; 240: n_truth=0, hit03=None
    assert s["models"]["m1"]["rungs"]["480"]["boxes"]["hit03"] == 1.0
    assert s["models"]["m1"]["rungs"]["240"]["boxes"]["hit03"] is None
    # retention should guard and return None (cur is None), not crash with TypeError
    assert s["models"]["m1"]["retention"]["hit03"]["240"] is None


def test_hit05_separate_greedy_match():
    """Regression test for FINDING 2: hit05 should use separate min_iou=0.5 match."""
    labels = {
        "a.jpg": [{"brand": "adidas", "box": [0, 0, 100, 100]}],
    }
    # det_a: conf=3, IoU=0.48 (fails min_iou=0.5 but passes 0.3)
    # det_b: conf=1, IoU=0.90 (passes min_iou=0.5)
    # With single match at 0.3 (old logic), high-conf det_a claims the truth; hit05 = 0.48 < 0.5.
    # With separate match at 0.5 (new logic), det_b claims the truth for hit05; hit05 = 1.0.
    det_a = {"brand": "adidas", "box": [0, 0, 48, 100], "conf": 3}  # IoU = 4800/10000 = 0.48
    det_b = {"brand": "adidas", "box": [0, 0, 90, 100], "conf": 1}  # IoU = 9000/10000 = 0.90
    raw = {"m1": [_mk_rows("m1", "a.jpg", 480, [det_a, det_b])]}
    models = [ModelCfg("m1", "openai", "m1")]
    s = sc.score_all(raw, labels, models, [480])
    r = s["models"]["m1"]["rungs"]["480"]
    # hit03 from det_a (IoU >= 0.3), hit05 from det_b (IoU >= 0.5)
    assert r["boxes"]["hit03"] == 1.0
    assert r["boxes"]["hit05"] == 1.0


def test_off_list_brand_detections_are_ignored():
    """A hallucinated brand string outside the truth universe must not create
    a new scored brand or an FP; the truth brands' scores stay unchanged."""
    labels = {"a.jpg": [{"brand": "adidas", "box": [0, 0, 100, 100],
                         "size": "large", "placement": "foreground",
                         "location": "chest"}]}
    det_good = {"brand": "adidas", "box": [10, 10, 90, 90], "size": "large",
                "placement": "foreground", "location": "chest", "conf": 3}
    det_offlist = {"brand": "herforder", "box": [0, 0, 50, 50], "size": "small",
                   "placement": "background", "location": "board", "conf": 1}
    raw = {"m1": [_mk_rows("m1", "a.jpg", 480, [det_good, det_offlist])]}
    s = sc.score_all(raw, labels, [ModelCfg("m1", "openai", "m1")], [480])
    r = s["models"]["m1"]["rungs"]["480"]
    assert "herforder" not in r["presence"]
    assert r["presence"]["adidas"]["f1"] == 1.0
    assert r["presence"]["_macro_f1"] == 1.0
    assert r["boxes"]["n_det"] == 1
