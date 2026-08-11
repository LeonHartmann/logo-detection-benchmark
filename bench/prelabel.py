"""Run ONE configured model over manifest images and write its detections into
data/labels/<id>.json as suggestions, so a human only renames/adjusts boxes
instead of drawing from scratch.

Reuses the same machinery as bench/run.py (config, prompt, provider, resize
ladder, detection parsing) but sequentially and for a single model, since a
prelabel pass is a one-off ~40-image job, not a full benchmark sweep.
"""
import json
import os
import time

from bench.prompt import build_prompt, load_refs, RETRY_SUFFIX
from bench.providers import make_provider, parse_detections
from bench.resize import derive, rungs_for

DEFAULT_MODEL = "qwen3-vl-plus"
RETRIES = 2  # a small backoff is enough here; this isn't the full run sweep


def pick_model(models, name=None):
    """The ModelCfg to use: `name` if given, else DEFAULT_MODEL, both must be
    enabled. Raises ValueError listing available (enabled) names otherwise."""
    enabled = {m.name: m for m in models if m.enabled}
    target = name or DEFAULT_MODEL
    if target not in enabled:
        available = ", ".join(sorted(enabled)) or "(none enabled)"
        raise ValueError(f"model {target!r} not found among enabled models: {available}")
    return enabled[target]


def _label_path(root, image_id):
    return os.path.join(root, "data", "labels", image_id + ".json")


def _skip_reason(path):
    """None if the image should be (re-)processed; a short reason otherwise.

    Never touch human work: skip if the label is marked done, or if it
    contains any box that isn't a suggestion (a human drew or edited it).
    A label file that exists but holds only suggestion boxes (or none) is
    fair game to refresh.
    """
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if d.get("done"):
        return "done"
    if any(not b.get("suggested") for b in d.get("boxes", [])):
        return "has human-authored boxes"
    return None


def prelabel(models, bench, brands, manifest, root, model_name=None, rung=None,
            provider_factory=make_provider, backoff_base=2.0):
    mcfg = pick_model(models, model_name)
    prov = provider_factory(mcfg, bench)

    from bench.config import load_reference_sheet
    refs = load_refs(brands, root, sheet_path=load_reference_sheet())
    prompt = build_prompt(brands, ref_labels=[lb for _, lb in refs])
    ref_bytes = [b for b, _ in refs]
    brand_names = {b["name"].strip().casefold() for b in brands}

    rungs_dir = os.path.join(root, "data", "rungs")
    os.makedirs(os.path.join(root, "data", "labels"), exist_ok=True)

    images = manifest["images"]
    for img in images:  # ensure the ladder exists before any API call
        derive(os.path.join(root, "data", "images", img["id"]), img["id"],
               rungs_dir, bench["rungs"], bench["jpeg_quality"])

    def attempt(text, images_payload):
        delay = backoff_base
        for i in range(RETRIES):
            try:
                return prov.call(text, images_payload), None
            except Exception as e:  # HTTP errors, timeouts
                if i == RETRIES - 1:
                    return None, f"{type(e).__name__}: {e}"
                time.sleep(delay)
                delay *= 2
        return None, "unreachable"

    stats = {"processed": 0, "skipped": 0, "failed": 0,
             "input_tokens": 0, "output_tokens": 0}

    for img in images:
        label_path = _label_path(root, img["id"])
        try:
            reason = _skip_reason(label_path)
            if reason:
                print(f"skip {img['id']}: {reason}")
                stats["skipped"] += 1
                continue

            target_rung = rung if rung is not None else max(
                rungs_for(img["native"][1], bench["rungs"]))
            with open(os.path.join(rungs_dir, str(target_rung), img["id"]), "rb") as f:
                target_bytes = f.read()
            payload = ref_bytes + [target_bytes]

            res, err = attempt(prompt, payload)
            if res is None:
                raise RuntimeError(err)
            dets = parse_detections(res.text)
            if dets is None:  # one re-ask with the bare-JSON reminder
                res2, err2 = attempt(prompt + RETRY_SUFFIX, payload)
                if res2 is None:
                    raise RuntimeError(err2)
                res = res2
                dets = parse_detections(res.text)
                if dets is None:
                    raise RuntimeError("response not parseable as detection JSON")

            stats["input_tokens"] += res.input_tokens
            stats["output_tokens"] += res.output_tokens

            boxes = [{"brand": d["brand"], "box": d["box"], "size": d["size"],
                     "placement": d["placement"], "location": d["location"],
                     "suggested": True}
                    for d in dets if d["brand"] in brand_names]

            json.dump({"image": img["id"], "labeler": f"prelabel:{mcfg.name}",
                      "boxes": boxes, "done": False},
                     open(label_path, "w"), indent=1)
            stats["processed"] += 1
        except Exception as e:
            print(f"WARNING: prelabel failed for {img['id']} ({label_path}): "
                 f"{type(e).__name__}: {e}")
            stats["failed"] += 1

    return stats
