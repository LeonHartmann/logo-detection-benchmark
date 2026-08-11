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
    # concurrency 1, single model/image, so the scripted replies map to work
    # items deterministically: work.sort()'s (image id, rung) key runs rung
    # 240 before rung 480 for this single image.
    bench_seq = {**BENCH, "concurrency": {"fake": 1}}
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider(["not json at all", GOOD, "garbage", "still garbage"])
    rn.run_benchmark(models, bench_seq, BRANDS, manifest, str(tmp_path),
                     provider_factory=lambda m, b: fp)
    rows = [json.loads(l) for l in
            open(tmp_path / "results" / "raw" / "fake-model.jsonl")]
    rows.sort(key=lambda r: r["rung"])
    assert rows[0]["retried"] is True and rows[0]["parse_ok"] is True   # 240: fail then good
    assert rows[1]["retried"] is True and rows[1]["parse_ok"] is False  # 480: fail twice
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
    """Verify the per-provider semaphore is actually enforced, not just pool sizing.

    Uses two models with different provider keys and concurrency caps:
    - fakeA: cap 1
    - fakeB: cap 4
    ThreadPoolExecutor gets max_workers=5, so if semaphore is a no-op,
    fakeA calls would overlap. We assert fakeA max_in_flight == 1 to prove
    the semaphore is enforced.
    """
    manifest = setup_repo(tmp_path, n_images=4)

    # Two models with different provider keys and caps
    models = [
        ModelCfg("m-a", "fakeA", "fake-1"),
        ModelCfg("m-b", "fakeB", "fake-2")
    ]

    providers = {}

    class ConcurrencyTrackingProvider:
        def __init__(self, key):
            self.provider_key = key
            self.lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0

        def call(self, text, images):
            with self.lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                time.sleep(0.02)  # 20ms to observe concurrency effects
                return CallResult(GOOD, 100, 10, 0.02)
            finally:
                with self.lock:
                    self.in_flight -= 1

    def provider_factory(mcfg, bench):
        key = mcfg.provider
        if key not in providers:
            providers[key] = ConcurrencyTrackingProvider(key)
        return providers[key]

    bench_cfg = {**BENCH, "concurrency": {"fakeA": 1, "fakeB": 4}, "rungs": [480]}
    # 4 images x 1 rung = 4 work items per model = 8 total
    stats = rn.run_benchmark(models, bench_cfg, BRANDS, manifest, str(tmp_path),
                             provider_factory=provider_factory, only_rungs=[480])

    # Verify work completed
    rows_a = [json.loads(l) for l in open(tmp_path / "results" / "raw" / "m-a.jsonl")]
    rows_b = [json.loads(l) for l in open(tmp_path / "results" / "raw" / "m-b.jsonl")]
    assert len(rows_a) == 4 and len(rows_b) == 4
    assert stats["done"] == 8

    # The critical assertion: fakeA's semaphore (cap 1) must be enforced
    # even though ThreadPoolExecutor has 5 workers available (1 + 4)
    # If semaphore is a no-op, with 4 fakeA items and 5 pool workers,
    # multiple fakeA calls would execute simultaneously -> max_in_flight >= 2
    assert providers["fakeA"].max_in_flight == 1, \
        f"fakeA semaphore not enforced: max_in_flight={providers['fakeA'].max_in_flight}, cap=1"

    # fakeB should respect its cap of 4 (typically much less due to serialization)
    assert providers["fakeB"].max_in_flight <= 4, \
        f"fakeB exceeded cap: max_in_flight={providers['fakeB'].max_in_flight}, cap=4"


def test_work_is_interleaved_across_models_for_parallel_provider_progress(tmp_path):
    """`work` is sorted by (image id, rung) so consecutive items belong to
    different models. Without that, a model-major `work` list means a
    shared ThreadPoolExecutor's FIFO queue dispatches one model's entire
    backlog to the pool before another model's items are even pulled off
    the queue -- serializing providers that could otherwise run in
    parallel. With pool size == sum(caps) == 2 (cap 1 per model here), the
    first two tasks the pool actually starts must belong to two different
    models; with the old model-major order they'd both be model A's."""
    manifest = setup_repo(tmp_path, n_images=4)
    models = [ModelCfg("m-a", "fakeA", "fake-1"), ModelCfg("m-b", "fakeB", "fake-2")]
    started = []
    lock = threading.Lock()

    class RecordingProvider:
        def __init__(self, key):
            self.provider_key = key

        def call(self, text, images):
            with lock:
                started.append(self.provider_key)
                hold = len(started) <= 2
            if hold:  # keep the first two pool slots busy until both are recorded
                time.sleep(0.05)
            return CallResult(GOOD, 100, 10, 0.01)

    providers = {}

    def provider_factory(mcfg, bench):
        return providers.setdefault(mcfg.provider, RecordingProvider(mcfg.provider))

    bench_cfg = {**BENCH, "concurrency": {"fakeA": 1, "fakeB": 1}, "rungs": [480]}
    rn.run_benchmark(models, bench_cfg, BRANDS, manifest, str(tmp_path),
                     provider_factory=provider_factory, only_rungs=[480])
    assert set(started[:2]) == {"fakeA", "fakeB"}


def test_failed_row_is_retried_and_appended_on_rerun(tmp_path):
    """An API-error row (error set) must not stick as permanently 'done':
    the next `bench run` should retry that (image, rung) and append a new
    row, leaving the old failed row in place (dedup happens at read time,
    in bench.cli.load_raw, not here)."""
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-model", "fake", "fake-1")]

    # First run: the provider errors on every attempt (max_retries=2) -> a
    # failed row with `error` set is written.
    fp1 = FakeProvider([RuntimeError("boom"), RuntimeError("boom")])
    stats1 = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                              only_rungs=[480], provider_factory=lambda m, b: fp1,
                              backoff_base=0)
    raw = tmp_path / "results" / "raw" / "fake-model.jsonl"
    rows1 = [json.loads(l) for l in open(raw)]
    assert len(rows1) == 1 and rows1[0]["error"] and stats1["failed"] == 1

    # Second run: the provider now succeeds. The failed (image, rung) must
    # be retried (not skipped), producing a second, successful row.
    fp2 = FakeProvider([])
    stats2 = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                              only_rungs=[480], provider_factory=lambda m, b: fp2,
                              backoff_base=0)
    rows2 = [json.loads(l) for l in open(raw)]
    assert stats2["done"] == 1 and stats2["skipped"] == 0
    assert len(rows2) == 2                       # old failed row still on disk, plus the new one
    assert rows2[0]["error"] and not rows2[0]["parse_ok"]
    assert rows2[1]["error"] is None and rows2[1]["parse_ok"] is True


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


def test_zoom_crop_returns_enlarged_jpeg():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (500, 400), "green").save(buf, "JPEG")
    out = rn._zoom_crop(buf.getvalue(), [100, 100, 500, 600])
    im = Image.open(io.BytesIO(out))
    assert im.format == "JPEG"
    # crop 100-500 of 1000 -> 200px wide of 500, x2 = 400; y 100-600 -> 200 of 400, x2 = 400
    assert im.size == (400, 400)


class ZoomFakeProvider:
    provider_key = "fake"
    def __init__(self, script):
        self.script = list(script)
        self.message_log = []
    def call_messages(self, messages):
        self.message_log.append([{"role": m["role"],
                                  "parts": [k for k, _ in m["parts"]]}
                                 for m in messages])
        return CallResult(self.script.pop(0), 100, 10, 0.01)
    def call(self, text, images):
        return self.call_messages([{"role": "user", "parts":
                                    [("image", i) for i in images] + [("text", text)]}])


def test_zoom_loop_requests_crop_then_finalizes(tmp_path):
    bench_seq = {**BENCH, "concurrency": {"fake": 1}}
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-zoom", "fake", "fake-1", tools=("zoom",))]
    fp = ZoomFakeProvider(['{"zoom": [0, 0, 500, 500]}', GOOD,
                           '{"zoom": [0, 0, 500, 500]}', GOOD])
    stats = rn.run_benchmark(models, bench_seq, BRANDS, manifest, str(tmp_path),
                             provider_factory=lambda m, b: fp)
    rows = [json.loads(l) for l in
            open(tmp_path / "results" / "raw" / "fake-zoom.jsonl")]
    assert stats["done"] == 2 and all(r["parse_ok"] for r in rows)
    assert all(r["zooms"] == 1 for r in rows)
    # tokens accumulate across the two turns of each call
    assert all(r["input_tokens"] == 200 and r["output_tokens"] == 20 for r in rows)
    # second request of the first conversation: user, assistant, user with a
    # new image part (the crop) plus text
    second = fp.message_log[1]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["parts"] == ["image", "text"]


def test_refs_condition_prepends_reference_images(tmp_path):
    from PIL import Image as PILImage
    manifest = setup_repo(tmp_path, n_images=1)
    (tmp_path / "data" / "refs").mkdir(parents=True)
    PILImage.new("RGB", (64, 64), "white").save(tmp_path / "data" / "refs" / "m.jpg")
    brands = [{"name": "adidas", "description": "x", "refs": []}]

    class CountingProvider:
        provider_key = "fake"
        def __init__(self):
            self.image_counts = []
        def call(self, text, images):
            self.image_counts.append(len(images))
            self.last_text = text
            return CallResult(GOOD, 1, 1, 0.01)

    fp = CountingProvider()
    models = [ModelCfg("fake-refs", "fake", "fake-1", refs=True)]
    rn.run_benchmark(models, {**BENCH, "concurrency": {"fake": 1}}, brands, manifest,
                     str(tmp_path), only_rungs=[480], provider_factory=lambda m, b: fp,
                     ref_sheet="data/refs/m.jpg")
    assert fp.image_counts == [2]  # reference sheet + target
    assert "reference sheet" in fp.last_text


def test_refs_model_skipped_when_no_refs_configured(tmp_path, capsys):
    manifest = setup_repo(tmp_path, n_images=1)
    fp = FakeProvider([])
    models = [ModelCfg("fake-refs", "fake", "fake-1", refs=True)]
    stats = rn.run_benchmark(models, {**BENCH, "concurrency": {"fake": 1}}, BRANDS,
                             manifest, str(tmp_path), provider_factory=lambda m, b: fp)
    assert stats["done"] == 0 and fp.calls == 0
    assert "skipping refs-condition models" in capsys.readouterr().out


def test_per_brand_condition_unions_focused_calls(tmp_path):
    manifest = setup_repo(tmp_path, n_images=1)
    brands2 = [{"name": "adidas", "description": "x", "refs": []},
               {"name": "delay", "description": "y", "refs": []}]

    class BrandProvider:
        provider_key = "fake"
        def __init__(self):
            self.texts = []
        def call(self, text, images):
            self.texts.append(text)
            if '"adidas"' in text:
                # includes an off-brand detection that must be filtered out
                return CallResult('{"detections":[{"brand":"adidas","box":[1,2,3,4]},'
                                  '{"brand":"delay","box":[5,6,7,8]}]}', 100, 10, 0.01)
            return CallResult('{"detections":[{"brand":"delay","box":[9,9,20,20]}]}',
                              100, 10, 0.01)

    fp = BrandProvider()
    models = [ModelCfg("fake-pb", "fake", "fake-1", per_brand=True)]
    stats = rn.run_benchmark(models, {**BENCH, "concurrency": {"fake": 1}}, brands2,
                             manifest, str(tmp_path), only_rungs=[480],
                             provider_factory=lambda m, b: fp)
    rows = [json.loads(l) for l in open(tmp_path / "results" / "raw" / "fake-pb.jsonl")]
    assert stats["done"] == 1 and rows[0]["parse_ok"]
    dets = rows[0]["detections"]
    # one adidas det (off-brand delay det from the adidas call filtered out),
    # one delay det from the delay call
    assert sorted(d["brand"] for d in dets) == ["adidas", "delay"]
    assert rows[0]["input_tokens"] == 200 and rows[0]["output_tokens"] == 20
    assert len(fp.texts) == 2
    assert '"adidas"' in fp.texts[0] and '"delay"' not in fp.texts[0].split("Rules:")[0].split("instance")[1]
