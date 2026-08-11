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
