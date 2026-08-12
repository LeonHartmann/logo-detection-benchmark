#!/usr/bin/env python3
"""Open-set local logo detection: Grounding DINO finds logo-like regions,
CLIP matches each region against a per-brand reference gallery. No per-brand
training; new brands are reference images in GALLERY. Writes standard raw
result rows so the stack is scored on the leaderboard like any API model.

Usage:
  python scripts/openset_scan.py              # full scan, resumable
  python scripts/openset_scan.py --calibrate  # print match scores on 6 frames
"""
import datetime
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from PIL import Image

from bench.cli import load_manifest
from bench.config import ROOT, load_bench
from bench.resize import rungs_for

NAME = "gdino-clip-openset"
DELAY = "/Users/leon/Coding/Rabona/delay-social-review"
GALLERY = {
    "adidas": [f"{DELAY}/data/qwen_test/refs/adidas.jpg",
               f"{DELAY}/deck/assets/adidas_logo.png"],
    "stripes": [f"{DELAY}/deck/assets/ann_stripes.jpg"],
    "dkh": [f"{DELAY}/data/qwen_test/refs/dkh.jpg",
            f"{DELAY}/deck/assets/dkh_logo.png"],
    "11teamsports": [f"{DELAY}/data/qwen_live/refs/11ts_backprint.jpg",
                     f"{DELAY}/data/qwen_live/refs/11ts_roundel_gold.jpg",
                     f"{DELAY}/data/qwen_live/refs/11ts_roundel_red.jpg",
                     f"{DELAY}/data/qwen_live/refs/11ts_wordmark_white.jpg",
                     f"{DELAY}/deck/assets/11teamsports_logo.png"],
    "delay": [f"{DELAY}/data/qwen_test/refs/delay.jpg",
              f"{DELAY}/data/qwen_test/refs/delay_jersey.jpg",
              f"{DELAY}/data/qwen_live/refs/delaytext_backprint.jpg",
              f"{DELAY}/data/qwen_live/refs/delaytext_whitetee.jpg",
              f"{DELAY}/deck/assets/delay_logo.png"],
    # negative gallery: crops that LOOK like our brands but are not
    "_negative": [f"{DELAY}/data/qwen_live/refs/neg_tjk_board.jpg",
                  f"{DELAY}/data/qwen_live/refs/neg_tsc_scoreboard.jpg"],
}
TEXT_HINTS = {
    "adidas": "the adidas logo, three slanted bars forming a triangle",
    "stripes": "three parallel stripes running along a sports jersey sleeve",
    "dkh": "the wordmark Deutsche Krebshilfe printed on a shirt",
    "11teamsports": "the 11teamsports logo, the number 11 in a circle",
    "delay": "the Delay Sports club crest, a round badge with a gate and ball",
}
GDINO_PROMPT = ("a brand logo. a club crest badge. a small logo on clothing. "
                "a scoreboard logo. printed text on a shirt.")
BOX_THRESHOLD = 0.25
SIM_THRESHOLD = 0.62    # min cosine to accept a brand match (CLIP floor is ~0.55)
MARGIN = 0.03           # best brand must beat runner-up by this much
NMS_IOU = 0.45          # dedup overlapping GDINO candidates

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        return bg
    return im.convert("RGB")


def main():
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                              CLIPModel, CLIPProcessor)
    calibrate = "--calibrate" in sys.argv
    print(f"device {DEVICE}; loading models...")
    gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(DEVICE).eval()
    cl_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()

    @torch.no_grad()
    def embed_images(ims):
        inp = cl_proc(images=ims, return_tensors="pt").to(DEVICE)
        v = cl.get_image_features(**inp)
        if not isinstance(v, torch.Tensor):  # transformers 5.x: already-projected embeds
            v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)

    @torch.no_grad()
    def embed_texts(texts):
        inp = cl_proc(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        v = cl.get_text_features(**inp)
        if not isinstance(v, torch.Tensor):
            v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)

    # gallery embeddings: per brand a stack of image embs (+ one text emb)
    bank, bank_labels = [], []
    for brand, paths in GALLERY.items():
        ims = [load_rgb(p) for p in paths if os.path.exists(p)]
        if not ims:
            print(f"warning: no gallery images for {brand}")
            continue
        embs = embed_images(ims)
        for e in embs:
            bank.append(e)
            bank_labels.append(brand)
    for brand, hint in TEXT_HINTS.items():
        bank.append(embed_texts([hint])[0])
        bank_labels.append(brand)
    bank_t = torch.stack(bank)
    print(f"gallery: {len(bank_labels)} entries for "
          f"{sorted(set(l for l in bank_labels if not l.startswith('_')))}")

    def _nms(boxes, scores):
        """Greedy dedup: drop boxes overlapping a higher-scored box."""
        order = sorted(range(len(boxes)), key=lambda i: -float(scores[i]))
        keep = []
        def biou(a, b):
            ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
            ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua if ua else 0
        for i in order:
            bi = [float(v) for v in boxes[i]]
            if all(biou(bi, [float(v) for v in boxes[j]]) < NMS_IOU for j in keep):
                keep.append(i)
        return [boxes[i] for i in keep], [scores[i] for i in keep]

    @torch.no_grad()
    def detect(im):
        inp = gd_proc(images=im, text=GDINO_PROMPT, return_tensors="pt").to(DEVICE)
        out = gd(**inp)
        res = gd_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=BOX_THRESHOLD, text_threshold=0.2,
            target_sizes=[im.size[::-1]])[0]
        return _nms(res["boxes"].cpu(), res["scores"].cpu())

    def classify(im, boxes, scores, verbose=False):
        dets = []
        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = [float(v) for v in box]
            # pad 8 percent, clamp
            pw, ph = (x1 - x0) * 0.08, (y1 - y0) * 0.08
            crop = im.crop((max(0, x0 - pw), max(0, y0 - ph),
                            min(im.width, x1 + pw), min(im.height, y1 + ph)))
            if crop.width < 8 or crop.height < 8:
                continue
            emb = embed_images([crop])[0]
            sims = bank_t @ emb
            # per-brand max similarity
            per_brand = {}
            for s, lb in zip(sims.tolist(), bank_labels):
                per_brand[lb] = max(per_brand.get(lb, -1), s)
            ranked = sorted(per_brand.items(), key=lambda kv: -kv[1])
            best, second = ranked[0], ranked[1] if len(ranked) > 1 else ("", -1)
            if verbose:
                print(f"    box {[round(v) for v in (x0, y0, x1, y1)]} "
                      f"gd {score:.2f} -> {best[0]} {best[1]:.3f} "
                      f"(2nd {second[0]} {second[1]:.3f})")
            if best[0].startswith("_") or best[1] < SIM_THRESHOLD \
                    or best[1] - second[1] < MARGIN:
                continue
            nb = [max(0, min(1000, round(v))) for v in
                  (x0 / im.width * 1000, y0 / im.height * 1000,
                   x1 / im.width * 1000, y1 / im.height * 1000)]
            area = (nb[2] - nb[0]) * (nb[3] - nb[1]) / 1e6
            size = "large" if area > 0.05 else ("medium" if area > 0.008 else "small")
            conf = 3 if float(score) > 0.45 else (2 if float(score) > 0.33 else 1)
            dets.append({"brand": best[0], "box": nb, "size": size,
                         "placement": "foreground", "location": "other",
                         "conf": conf})
        return dets

    manifest = load_manifest(ROOT)
    bench = load_bench()
    rungs = bench["rungs"]
    out_path = os.path.join(ROOT, "results", "raw", NAME + ".jsonl")
    done = set()
    if os.path.exists(out_path) and not calibrate:
        for l in open(out_path):
            try:
                r = json.loads(l)
                if not r.get("error"):
                    done.add((r["image"], r["rung"]))
            except (json.JSONDecodeError, KeyError):
                pass
    images = manifest["images"][:6] if calibrate else manifest["images"]
    n = 0
    f = None if calibrate else open(out_path, "a")
    for img in images:
        for rung in ([1080] if calibrate else rungs_for(img["native"][1], rungs)):
            if (img["id"], rung) in done:
                continue
            src = os.path.join(ROOT, "data", "rungs", str(rung), img["id"])
            if not os.path.exists(src):
                continue
            im = Image.open(src).convert("RGB")
            t0 = time.time()
            boxes, scores = detect(im)
            if calibrate:
                print(f"{img['id']} @{rung}p: {len(boxes)} candidate regions")
            dets = classify(im, boxes, scores, verbose=calibrate)
            lat = time.time() - t0
            if not calibrate:
                row = {"image": img["id"], "rung": rung, "model": NAME,
                       "detections": dets, "parse_ok": True, "retried": False,
                       "latency_s": round(lat, 3), "input_tokens": 0,
                       "output_tokens": 0, "error": None,
                       "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                f.write(json.dumps(row) + "\n")
                f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  {n} keys scanned")
    if f:
        f.close()
    print(f"done: {n} keys scanned")


if __name__ == "__main__":
    main()
