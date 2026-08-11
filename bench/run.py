"""Fan out model x image x rung calls; resumable JSONL output per model."""
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import io

from PIL import Image

from bench.prompt import build_prompt, load_refs, RETRY_SUFFIX, ZOOM_LIMIT
from bench.providers import make_provider, parse_detections, parse_zoom
from bench.resize import derive, rungs_for


def _zoom_crop(target_bytes, box, quality=85):
    """An enlarged (2x) crop of the SAME rung image the model is analyzing.

    Zoom never reveals pixels the rung does not contain: it crops the rung
    image and upscales, so the resolution-robustness axis stays honest."""
    im = Image.open(io.BytesIO(target_bytes)).convert("RGB")
    px = [box[0] * im.width // 1000, box[1] * im.height // 1000,
          max(box[2] * im.width // 1000, box[0] * im.width // 1000 + 8),
          max(box[3] * im.height // 1000, box[1] * im.height // 1000 + 8)]
    crop = im.crop(px)
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _load_done(path):
    """(image, rung) pairs that don't need another API call.

    A row with a truthy `error` means the provider call itself failed (after
    the runner's own retries) — that's retried on the next `bench run`, not
    treated as permanently done. A row with `error: null` but `parse_ok:
    false` means the call succeeded but the model's output didn't parse;
    that stays done (retrying an API that answered, just badly, wastes
    budget without a code fix).
    """
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue
                done.add((r["image"], r["rung"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _per_brand_worker(prov, brands, refs_by_brand, use_refs, target, row, attempt):
    """One focused call per brand (optionally with only that brand's reference
    images); detections are unioned. Tokens and latency accumulate; parse_ok
    is true only if every per-brand call parsed."""
    dets_all = []
    in_tok = out_tok = 0
    lat = 0.0
    all_ok = True
    for b in brands:
        refs = refs_by_brand.get(b["name"], []) if use_refs else []
        text = build_prompt([b], ref_labels=[lb for _, lb in refs])
        payload = [bt for bt, _ in refs] + [target]
        res, err = attempt(prov, text, payload)
        if res is not None:
            dets = parse_detections(res.text)
            if dets is None:
                row["retried"] = True
                res2, err2 = attempt(prov, text + RETRY_SUFFIX, payload)
                if res2 is not None:
                    dets = parse_detections(res2.text)
                    res = res2
                err = err2
        if res is None:
            row["error"] = err
            return
        in_tok += res.input_tokens
        out_tok += res.output_tokens
        lat += res.latency_s
        if dets is None:
            all_ok = False
        else:
            dets_all.extend(d for d in dets if d["brand"] == b["name"])
    row.update(latency_s=round(lat, 3), input_tokens=in_tok, output_tokens=out_tok)
    if all_ok:
        row.update(detections=dets_all, parse_ok=True)


def _zoom_worker(prov, m, text, payload, target, row, attempt_msgs):
    """Multi-turn zoom conversation. The model may request up to ZOOM_LIMIT
    enlarged crops of the rung image before giving its final detections.
    Token usage and latency accumulate across turns; row records the zoom
    count so scoring and cost stay honest."""
    messages = [{"role": "user",
                 "parts": [("image", b) for b in payload] + [("text", text)]}]
    zooms = in_tok = out_tok = 0
    lat = 0.0
    dets = None
    while True:
        res, err = attempt_msgs(prov, messages)
        if res is None:
            if row["detections"] is None:
                row["error"] = err
            break
        in_tok += res.input_tokens
        out_tok += res.output_tokens
        lat += res.latency_s
        z = parse_zoom(res.text)
        if z is not None and zooms < ZOOM_LIMIT:
            zooms += 1
            crop = _zoom_crop(target, z)
            messages.append({"role": "assistant", "parts": [("text", res.text)]})
            messages.append({"role": "user", "parts": [
                ("image", crop),
                ("text", f"Zoom result of {z}, enlarged 2x from the same image. "
                         f"{ZOOM_LIMIT - zooms} zooms remaining. Respond with the "
                         "final detections JSON (coordinates over the FULL original "
                         "target image), or one more zoom request.")]})
            continue
        dets = parse_detections(res.text)
        if dets is None and not row["retried"]:
            row["retried"] = True
            messages.append({"role": "assistant", "parts": [("text", res.text)]})
            messages.append({"role": "user", "parts": [("text", RETRY_SUFFIX.strip())]})
            continue
        break
    row.update(latency_s=round(lat, 3), input_tokens=in_tok,
               output_tokens=out_tok, zooms=zooms)
    if dets is not None:
        row.update(detections=dets, parse_ok=True)


def run_benchmark(models, bench, brands, manifest, root, only_models=None,
                  only_rungs=None, limit_images=None, provider_factory=make_provider,
                  backoff_base=2.0, ref_sheet=None):
    rungs = only_rungs or bench["rungs"]
    images = manifest["images"][:limit_images] if limit_images else manifest["images"]
    raw_dir = os.path.join(root, "results", "raw")
    rungs_dir = os.path.join(root, "data", "rungs")
    os.makedirs(raw_dir, exist_ok=True)

    refs = load_refs(brands, root, sheet_path=ref_sheet)
    ref_bytes = [b for b, _ in refs]
    refs_by_brand = {b["name"]: load_refs([b], root) for b in brands}
    ref_labels = [lb for _, lb in refs]
    prompts = {(r, z): build_prompt(brands, ref_labels if r else (), zoom=z)
               for r in (False, True) for z in (False, True)}
    prompt = prompts[(False, False)]  # plain condition, used by the retry path

    for img in images:  # ensure ladder exists before any API call
        derive(os.path.join(root, "data", "images", img["id"]), img["id"],
               rungs_dir, rungs, bench["jpeg_quality"])

    # explicitly named models run even when their row is disabled: naming a
    # condition row on the command line is a deliberate act
    active = [m for m in models
              if (m.name in only_models if only_models else m.enabled)]
    skipped_refs = [m.name for m in active if m.refs and not ref_bytes]
    if skipped_refs:
        print(f"skipping refs-condition models (no reference images configured): "
              f"{', '.join(skipped_refs)}")
        active = [m for m in active if not (m.refs and not ref_bytes)]
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
    # `work` is built model-major (all of model A's items, then all of model
    # B's, ...). With a single shared ThreadPoolExecutor, ex.map consumes it
    # roughly in order, so the pool spends its early submissions entirely on
    # model A -- capped at model A's provider concurrency -- before model B's
    # work is even seen. Sorting by (image id, rung) interleaves models so
    # consecutive items belong to different models, letting distinct
    # providers' semaphores fill in parallel instead of one at a time.
    work.sort(key=lambda t: (t[1]["id"], t[2]))

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

    def attempt_msgs(prov, messages):
        delay = backoff_base
        for i in range(bench["max_retries"]):
            try:
                return prov.call_messages(messages), None
            except Exception as e:
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
        use_refs = bool(m.refs and ref_bytes)
        zoom = "zoom" in (m.tools or ())
        text = prompts[(use_refs, zoom)]
        try:
            with open(os.path.join(root, "data", "rungs", str(rung), img["id"]), "rb") as f:
                target = f.read()
            payload = (ref_bytes if use_refs else []) + [target]
            with sems[prov.provider_key]:
                if m.per_brand:
                    _per_brand_worker(prov, brands, refs_by_brand, m.refs,
                                      target, row, attempt)
                elif zoom:
                    _zoom_worker(prov, m, text, payload, target, row, attempt_msgs)
                else:
                    res, err = attempt(prov, text, payload)
                    if res is not None:
                        dets = parse_detections(res.text)
                        if dets is None:  # one re-ask with the bare-JSON reminder, recorded
                            row["retried"] = True
                            res2, err2 = attempt(prov, text + RETRY_SUFFIX, payload)
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
