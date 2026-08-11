import json
import bench.cli as cli


def test_helpers_load(tmp_path):
    (tmp_path / "data" / "labels").mkdir(parents=True)
    (tmp_path / "results" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(
        {"images": [{"id": "a.jpg", "native": [854, 480], "stratum": "n", "source": {}}]}))
    (tmp_path / "data" / "labels" / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "boxes": [], "done": True}))
    (tmp_path / "data" / "labels" / "b.jpg.json").write_text(json.dumps(
        {"image": "b.jpg", "boxes": [], "done": False}))
    (tmp_path / "results" / "raw" / "m1.jsonl").write_text(
        json.dumps({"image": "a.jpg", "rung": 480, "model": "m1", "detections": [],
                    "parse_ok": True, "retried": False, "latency_s": 1,
                    "input_tokens": 1, "output_tokens": 1, "error": None, "ts": "t"}) + "\n")
    assert cli.load_manifest(str(tmp_path))["images"][0]["id"] == "a.jpg"
    labels = cli.load_labels(str(tmp_path))
    assert list(labels) == ["a.jpg"]          # not-done label excluded
    raw = cli.load_raw(str(tmp_path))
    assert list(raw) == ["m1"] and len(raw["m1"]) == 1


def test_parser_has_all_subcommands():
    parser = cli.build_parser()
    subs = parser._subparsers._group_actions[0].choices
    assert set(subs) == {"ladder", "run", "score", "report", "serve", "apply-reviews"}
