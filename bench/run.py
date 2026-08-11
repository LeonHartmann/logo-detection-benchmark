"""Fan out model x image x rung calls; resumable JSONL output per model."""
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from bench.prompt import build_prompt, load_refs, RETRY_SUFFIX
from bench.providers import make_provider, parse_detections
from bench.resize import derive, rungs_for


def _load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["image"], r["rung"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_benchmark(models, bench, brands, manifest, root, only_models=None,
                  only_rungs=None, limit_images=None, provider_factory=make_provider,
                  backoff_base=2.0):
    rungs = only_rungs or bench["rungs"]
    images = manifest["images"][:limit_images] if limit_images else manifest["images"]
    raw_dir = os.path.join(root, "results", "raw")
    rungs_dir = os.path.join(root, "data", "rungs")
    os.makedirs(raw_dir, exist_ok=True)

    refs = load_refs(brands, root)
    prompt = build_prompt(brands, n_refs=len(refs))
    ref_bytes = [b for b, _ in refs]

    for img in images:  # ensure ladder exists before any API call
        derive(os.path.join(root, "data", "images", img["id"]), img["id"],
               rungs_dir, rungs, bench["jpeg_quality"])

    active = [m for m in models
              if m.enabled and (not only_models or m.name in only_models)]
    sems = {}
    for m in active:
        prov = provider_factory(m, bench)
        key = prov.provider_key
        sems.setdefault(key, threading.Semaphore(bench["concurrency"].get(key, 4)))
        m._prov = prov  # attach for the worker

    stats = {"done": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    files = {m.name: open(os.path.join(raw_dir, m.name + ".jsonl"), "a") for m in active}
    work = []
    for m in active:
        done = _load_done(os.path.join(raw_dir, m.name + ".jsonl"))
        for img in images:
            for rung in rungs_for(img["native"][1], rungs):
                if (img["id"], rung) in done:
                    stats["skipped"] += 1
                else:
                    work.append((m, img, rung))

    def attempt(prov, text, images_payload):
        delay = backoff_base
        for i in range(bench["max_retries"]):
            try:
                return prov.call(text, images_payload), None
            except Exception as e:  # HTTP errors, timeouts
                if i == bench["max_retries"] - 1:
                    return None, f"{type(e).__name__}: {e}"
                time.sleep(delay)
                delay *= 2
        return None, "unreachable"

    def worker(item):
        m, img, rung = item
        prov = m._prov
        row = {"image": img["id"], "rung": rung, "model": m.name, "detections": None,
               "parse_ok": False, "retried": False, "latency_s": 0.0,
               "input_tokens": 0, "output_tokens": 0, "error": None,
               "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            with open(os.path.join(root, "data", "rungs", str(rung), img["id"]), "rb") as f:
                target = f.read()
            payload = ref_bytes + [target]
            with sems[prov.provider_key]:
                res, err = attempt(prov, prompt, payload)
                if res is not None:
                    dets = parse_detections(res.text)
                    if dets is None:  # one re-ask with the bare-JSON reminder, recorded
                        row["retried"] = True
                        res2, err2 = attempt(prov, prompt + RETRY_SUFFIX, payload)
                        if res2 is not None:
                            dets = parse_detections(res2.text)
                            res = res2
                        err = err2
                    row.update(latency_s=round(res.latency_s, 3),
                               input_tokens=res.input_tokens,
                               output_tokens=res.output_tokens)
                    if dets is not None:
                        row.update(detections=dets, parse_ok=True)
                if err and row["detections"] is None:
                    row["error"] = err
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        with lock:
            files[m.name].write(json.dumps(row) + "\n")
            files[m.name].flush()
            stats["failed" if row["error"] else "done"] += 1
            n = stats["done"] + stats["failed"]
            if n % 25 == 0:
                print(f"  {n}/{len(work)} calls done")

    try:
        if work:
            with ThreadPoolExecutor(max_workers=sum(
                    bench["concurrency"].get(k, 4) for k in sems)) as ex:
                list(ex.map(worker, work))
    finally:
        for f in files.values():
            f.close()
    return stats
