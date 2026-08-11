import glob
import json
import os
import shutil
import yaml
from pathlib import Path
from PIL import Image

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
                    "input_tokens": 1, "output_tokens": 1, "error": None, "ts": "t"}) + "\n"
        + "not valid json at all, e.g. a truncated write\n"      # json.JSONDecodeError
        + json.dumps({"image": "c.jpg", "model": "m1"}) + "\n")  # KeyError: no "rung"
    assert cli.load_manifest(str(tmp_path))["images"][0]["id"] == "a.jpg"
    labels = cli.load_labels(str(tmp_path))
    assert list(labels) == ["a.jpg"]          # not-done label excluded
    raw = cli.load_raw(str(tmp_path))
    assert list(raw) == ["m1"] and len(raw["m1"]) == 1  # garbage line skipped, not raised


def test_load_raw_dedupes_by_image_rung_keeping_last(tmp_path):
    """A retried (image, rung) pair leaves two rows in the same model's
    JSONL (an old failed attempt, then a later successful one). load_raw
    must collapse those to one row per (image, rung) -- the last one on
    disk -- so scoring doesn't double-count a frame."""
    (tmp_path / "results" / "raw").mkdir(parents=True)
    failed = {"image": "a.jpg", "rung": 480, "model": "m1", "detections": None,
              "parse_ok": False, "retried": False, "latency_s": 0.0,
              "input_tokens": 0, "output_tokens": 0, "error": "boom", "ts": "t1"}
    retried_ok = {"image": "a.jpg", "rung": 480, "model": "m1", "detections": [],
                  "parse_ok": True, "retried": False, "latency_s": 1.0,
                  "input_tokens": 1, "output_tokens": 1, "error": None, "ts": "t2"}
    other = {"image": "b.jpg", "rung": 240, "model": "m1", "detections": [],
              "parse_ok": True, "retried": False, "latency_s": 1.0,
              "input_tokens": 1, "output_tokens": 1, "error": None, "ts": "t3"}
    with open(tmp_path / "results" / "raw" / "m1.jsonl", "w") as f:
        for row in (failed, retried_ok, other):
            f.write(json.dumps(row) + "\n")

    raw = cli.load_raw(str(tmp_path))
    rows = raw["m1"]
    assert len(rows) == 2                     # one per (image, rung), not three
    by_key = {(r["image"], r["rung"]): r for r in rows}
    assert by_key[("a.jpg", 480)]["ts"] == "t2"       # the LAST row for that key wins
    assert by_key[("a.jpg", 480)]["error"] is None
    assert by_key[("b.jpg", 240)]["ts"] == "t3"


def test_score_warns_about_images_missing_done_labels(tmp_path, monkeypatch, capsys):
    """An image that shows up in raw model output but has no completed
    ("done") truth label scores as if it had zero truth boxes -- silently,
    unless `score` warns about it. This is the truth-hole `bench score`
    should surface, without changing what scores.json actually contains."""
    (tmp_path / "data" / "labels").mkdir(parents=True)
    (tmp_path / "results" / "raw").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)

    (tmp_path / "configs" / "benchmark.yaml").write_text(yaml.dump({
        "rungs": [480], "jpeg_quality": 85, "max_tokens": 100, "timeout_s": 5,
        "max_retries": 2, "concurrency": {"openai": 1}, "brands_file": "brands/delay.yaml"}))
    (tmp_path / "configs" / "models.yaml").write_text(yaml.dump({
        "models": [{"name": "m1", "provider": "openai", "model": "m1"}]}))

    # a.jpg has a completed label; b.jpg only appears in the raw results.
    (tmp_path / "data" / "labels" / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "boxes": [], "done": True}))
    row = {"model": "m1", "detections": [], "parse_ok": True, "retried": False,
           "latency_s": 1, "input_tokens": 1, "output_tokens": 1, "error": None, "ts": "t"}
    (tmp_path / "results" / "raw" / "m1.jsonl").write_text(
        json.dumps({**row, "image": "a.jpg", "rung": 480}) + "\n"
        + json.dumps({**row, "image": "b.jpg", "rung": 480}) + "\n")

    monkeypatch.setattr("bench.config.ROOT", str(tmp_path))
    cli.main(["score"])

    out = capsys.readouterr().out
    assert "WARNING" in out and "1 images" in out and "b.jpg" in out
    assert "a.jpg" not in out.split("WARNING", 1)[1]  # only the labelless image is named
    scores = json.loads((tmp_path / "results" / "scores.json").read_text())
    assert scores["n_images"] == 1  # scoring itself is unchanged: still just the done label


def test_parser_has_all_subcommands():
    parser = cli.build_parser()
    subs = parser._subparsers._group_actions[0].choices
    assert set(subs) == {"ladder", "manifest", "run", "score", "report", "serve", "apply-reviews"}


def test_ladder_writes_brands_json(tmp_path, monkeypatch):
    # Create minimal directory structure
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "rungs").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "brands").mkdir(parents=True)

    # Create a small JPEG image
    img = Image.new("RGB", (854, 480), color="red")
    img.save(tmp_path / "data" / "images" / "test.jpg", "JPEG")

    # Create manifest.json
    manifest = {
        "images": [
            {"id": "test.jpg", "native": [854, 480], "stratum": "n", "source": {}}
        ]
    }
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest))

    # Create configs/benchmark.yaml
    benchmark_config = {
        "rungs": [1080, 720, 480, 240, 144],
        "jpeg_quality": 85,
        "max_tokens": 2000,
        "timeout_s": 180,
        "max_retries": 4,
        "concurrency": {
            "dashscope": 8,
            "openai": 8,
            "openrouter": 8,
            "anthropic": 8
        },
        "brands_file": "brands/delay.yaml"
    }
    (tmp_path / "configs" / "benchmark.yaml").write_text(yaml.dump(benchmark_config))

    # Create configs/models.yaml with minimal content
    models_config = {
        "models": []
    }
    (tmp_path / "configs" / "models.yaml").write_text(yaml.dump(models_config))

    # Create brands/delay.yaml
    brands_config = {
        "brands": [
            {"name": "adidas", "description": "adidas logo", "refs": []},
            {"name": "stripes", "description": "stripes", "refs": []},
            {"name": "dkh", "description": "dkh", "refs": []},
            {"name": "11teamsports", "description": "11teamsports", "refs": []},
            {"name": "delay", "description": "delay", "refs": []}
        ]
    }
    (tmp_path / "brands" / "delay.yaml").write_text(yaml.dump(brands_config))

    # Monkeypatch config.ROOT to point to tmp_path
    monkeypatch.setattr("bench.config.ROOT", str(tmp_path))

    # Run ladder command
    cli.main(["ladder"])

    # Assert brands.json was created
    brands_json_path = tmp_path / "data" / "brands.json"
    assert brands_json_path.exists()

    # Assert brands.json contains expected brands in order
    brands = json.loads(brands_json_path.read_text())
    assert brands == ["adidas", "stripes", "dkh", "11teamsports", "delay"]

    # Assert at least one rung file exists (in subdirectories like rungs/480/test.jpg)
    rung_files = glob.glob(str(tmp_path / "data" / "rungs" / "*" / "*.jpg"))
    assert len(rung_files) > 0


def test_manifest_merges_kept_entries_and_drops_missing_ones(tmp_path, monkeypatch, capsys):
    """`bench manifest` scans data/images/, merges by filename against any
    existing manifest.json: kept files retain their stratum/source, new
    files get stratum "unlabeled" with an empty source, and entries for
    files no longer on disk are dropped (and printed)."""
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)

    # "a.jpg" is already labeled in the existing manifest -- its native
    # size there is deliberately wrong, to prove the command re-reads the
    # real size from disk rather than trusting the old entry.
    Image.new("RGB", (640, 360), color="blue").save(
        tmp_path / "data" / "images" / "a.jpg", "JPEG")
    # "b.png" is new -- not present in the existing manifest at all.
    Image.new("RGB", (100, 200), color="green").save(
        tmp_path / "data" / "images" / "b.png", "PNG")

    existing_manifest = {
        "images": [
            {"id": "a.jpg", "native": [999, 999], "stratum": "busy",
             "source": {"type": "custom", "note": "existing"}},
            {"id": "missing.jpg", "native": [1, 1], "stratum": "n", "source": {}},
        ]
    }
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(existing_manifest))

    (tmp_path / "configs" / "benchmark.yaml").write_text(yaml.dump({
        "rungs": [480], "jpeg_quality": 85, "max_tokens": 100, "timeout_s": 5,
        "max_retries": 2, "concurrency": {"openai": 1}, "brands_file": "brands/delay.yaml"}))

    monkeypatch.setattr("bench.config.ROOT", str(tmp_path))

    cli.main(["manifest"])

    out = capsys.readouterr().out
    assert "missing.jpg" in out                      # dropped entry named
    assert "2 images (1 new, 1 kept, 1 dropped)" in out

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    ids = [img["id"] for img in manifest["images"]]
    assert ids == ["a.jpg", "b.png"]                  # sorted by id

    by_id = {img["id"]: img for img in manifest["images"]}
    assert by_id["a.jpg"]["native"] == [640, 360]     # re-read from disk, not [999, 999]
    assert by_id["a.jpg"]["stratum"] == "busy"         # kept untouched
    assert by_id["a.jpg"]["source"] == {"type": "custom", "note": "existing"}

    assert by_id["b.png"]["native"] == [100, 200]
    assert by_id["b.png"]["stratum"] == "unlabeled"    # new file default
    assert by_id["b.png"]["source"] == {}
