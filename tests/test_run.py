import json
import os
import threading
import time
from PIL import Image
import bench.run as rn
from bench.config import ModelCfg
from bench.providers import CallResult

BENCH = {"rungs": [480, 240], "jpeg_quality": 85, "max_tokens": 100, "timeout_s": 5,
         "max_retries": 2, "concurrency": {"fake": 2}}
BRANDS = [{"name": "adidas", "description": "x", "refs": []}]
GOOD = '{"detections":[{"brand":"adidas","box":[10,10,20,20],"size":"small","placement":"foreground","location":"chest","conf":3}]}'


class FakeProvider:
    provider_key = "fake"
    def __init__(self, script):
        self.script, self.calls = list(script), 0
    def call(self, text, images):
        self.calls += 1
        reply = self.script.pop(0) if self.script else GOOD
        if isinstance(reply, Exception):
            raise reply
        return CallResult(reply, 100, 10, 0.01)


def setup_repo(tmp_path, n_images=2):
    (tmp_path / "data" / "images").mkdir(parents=True)
    images = []
    for i in range(n_images):
        name = f"img{i}.jpg"
        Image.new("RGB", (854, 480), "green").save(tmp_path / "data" / "images" / name)
        images.append({"id": name, "native": [854, 480], "stratum": "normal", "source": {}})
    return {"images": images}


def test_run_writes_rows_and_resumes(tmp_path):
    manifest = setup_repo(tmp_path)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider([])
    stats = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                             provider_factory=lambda m, b: fp)
    raw = tmp_path / "results" / "raw" / "fake-model.jsonl"
    rows = [json.loads(l) for l in open(raw)]
    assert len(rows) == 4  # 2 images x 2 rungs
    assert stats["done"] == 4
    assert all(r["parse_ok"] and r["detections"] for r in rows)
    assert {(r["image"], r["rung"]) for r in rows} == {
        ("img0.jpg", 480), ("img0.jpg", 240), ("img1.jpg", 480), ("img1.jpg", 240)}
    # resume: nothing new is called
    before = fp.calls
    stats2 = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                              provider_factory=lambda m, b: fp)
    assert fp.calls == before and stats2["skipped"] == 4


def test_parse_failure_triggers_single_retry(tmp_path):
    # concurrency 1 so the scripted replies map to work items deterministically
    bench_seq = {**BENCH, "concurrency": {"fake": 1}}
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider(["not json at all", GOOD, "garbage", "still garbage"])
    rn.run_benchmark(models, bench_seq, BRANDS, manifest, str(tmp_path),
                     provider_factory=lambda m, b: fp)
    rows = [json.loads(l) for l in
            open(tmp_path / "results" / "raw" / "fake-model.jsonl")]
    rows.sort(key=lambda r: r["rung"], reverse=True)
    assert rows[0]["retried"] is True and rows[0]["parse_ok"] is True   # 480: fail then good
    assert rows[1]["retried"] is True and rows[1]["parse_ok"] is False  # 240: fail twice
    assert rows[1]["detections"] is None


def test_api_error_is_recorded_after_retries(tmp_path):
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider([RuntimeError("boom"), RuntimeError("boom"),
                       RuntimeError("boom"), RuntimeError("boom")])
    stats = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                             only_rungs=[480], provider_factory=lambda m, b: fp,
                             backoff_base=0)
    rows = [json.loads(l) for l in
            open(tmp_path / "results" / "raw" / "fake-model.jsonl")]
    assert len(rows) == 1 and rows[0]["error"] and rows[0]["detections"] is None
    assert stats["failed"] == 1


def test_concurrency_cap_is_enforced(tmp_path):
    """Verify the per-provider concurrency semaphore is actually enforced."""
    manifest = setup_repo(tmp_path, n_images=3)
    models = [ModelCfg("fake-model", "fake", "fake-1")]

    class ConcurrencyTrackingProvider:
        provider_key = "fake"
        def __init__(self):
            self.lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0
        def call(self, text, images):
            with self.lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                time.sleep(0.01)  # brief hold to observe concurrency
                return CallResult(GOOD, 100, 10, 0.01)
            finally:
                with self.lock:
                    self.in_flight -= 1

    cp = ConcurrencyTrackingProvider()
    bench_cfg = {**BENCH, "concurrency": {"fake": 2}}
    stats = rn.run_benchmark(models, bench_cfg, BRANDS, manifest, str(tmp_path),
                             provider_factory=lambda m, b: cp)
    rows = [json.loads(l) for l in open(tmp_path / "results" / "raw" / "fake-model.jsonl")]
    assert len(rows) == 6  # 3 images x 2 rungs
    assert stats["done"] == 6
    assert cp.max_in_flight <= 2, f"max_in_flight {cp.max_in_flight} exceeds cap of 2"


def test_worker_file_error_doesnt_crash_run(tmp_path, monkeypatch):
    """Verify that worker-level failures (e.g., missing file) are caught and recorded."""
    import builtins
    manifest = setup_repo(tmp_path, n_images=2)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider([])

    original_open = builtins.open
    rung_read_calls = [0]

    def patched_open(file, *args, **kwargs):
        # Only fail when reading rung files (from data/rungs directory with "rb" mode)
        # Not during derive (which reads from data/images)
        if (len(args) > 0 and args[0] == "rb" and
            "/data/rungs/" in str(file) and "img0.jpg" in str(file)):
            rung_read_calls[0] += 1
            if rung_read_calls[0] == 1:
                raise FileNotFoundError(f"[Errno 2] No such file or directory: {file}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", patched_open)

    stats = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                             provider_factory=lambda m, b: fp)
    rows = [json.loads(l) for l in original_open(tmp_path / "results" / "raw" / "fake-model.jsonl")]

    # Should have 4 rows total (2 images x 2 rungs), with at least 1 error
    assert len(rows) == 4
    assert stats["failed"] >= 1
    assert stats["done"] + stats["failed"] == 4

    # The failed row should have an error message and no detections
    failed_rows = [r for r in rows if r["error"] is not None]
    assert len(failed_rows) >= 1
    for row in failed_rows:
        assert row["detections"] is None
        assert "FileNotFoundError" in row["error"] or "No such file" in row["error"]
