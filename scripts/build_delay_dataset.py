#!/usr/bin/env python3
"""Build the 40-frame Delay dataset for the benchmark.

Selection (stratified from the Delay repo's 140-frame bench set), target
distribution small=10/busy=10/normal=12/empty=8 split across the two
corpora via QUOTA below:
  YT (31): busy (truth >= 4 marks), normal, empty, small-logo (frames
           whose truth includes dkh or 11teamsports)
  IG (9 of a nominal 16): 6 dkh/small-text statics, 2 busy, 4 normal,
           4 empty -- the Delay repo's igimg population only actually
           supplies busy=0/normal=2/empty=1 of that, so the yt quota
           below absorbs the difference (see NOTE by QUOTA).
YT frames are re-extracted at 1080p from the source videos with ffmpeg
(only videos already present in the Delay repo are used; entries without
a local 1080p video are skipped in favor of the next candidate).
IG statics are copied at native resolution (must be >= 1080px tall,
smaller ones are skipped with a warning).
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys

from PIL import Image

SRC = "/Users/leon/Coding/Rabona/delay-social-review"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_IMG = os.path.join(ROOT, "data", "images")
SHEET_RE = re.compile(r"^ytsheet__(\d{8}_.+?)_s(\d+)__c(\d+)\.jpg$")
QUOTA = {"yt": {"busy": 10, "normal": 10, "empty": 7, "small": 4},
         "ig": {"small": 6, "busy": 2, "normal": 4, "empty": 4}}
# NOTE: yt quota rebalanced up from the brief's 8/8/4/4 (sum 24) to
# 10/10/7/4 (sum 31). The Delay repo's igimg population only supplies
# busy=0, normal=2, empty=1 candidates against its target busy=2/normal=4/
# empty=4 (small=6 is fully achievable) -- a fixed shortfall of 7 that no
# amount of video fetching can fix, since IG statics aren't derived from
# video. The yt quota absorbs exactly that shortfall in the same strata
# (busy +2, normal +2, empty +3) so the combined yt+ig totals still match
# the brief's original target distribution: small=10, busy=10, normal=12,
# empty=8, summing to 40.


def stratum(entry):
    t = entry.get("truth", {})
    n = sum(t.values())
    if "dkh" in t or "11teamsports" in t:
        return "small"
    if n >= 4:
        return "busy"
    if n == 0:
        return "empty"
    return "normal"


def yt_second(frame_id):
    m = SHEET_RE.match(frame_id)
    if not m:
        return None
    base, sheet, cell = m.group(1), int(m.group(2)), int(m.group(3))
    return base, ((sheet - 1) * 6 + (cell - 1)) * 5


def extract_yt(base, second, out_path):
    mp4 = os.path.join(SRC, "data", "qwen_res", "vid", base + ".mp4")
    if not os.path.exists(mp4):
        return False
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(second),
                        "-i", mp4, "-frames:v", "1", "-q:v", "2", "-y", out_path])
    return r.returncode == 0 and os.path.exists(out_path)


def main():
    rnd = random.Random(11)
    bench = json.load(open(os.path.join(SRC, "data", "qwen_bench", "bench_set.json")))
    rnd.shuffle(bench)
    os.makedirs(OUT_IMG, exist_ok=True)
    picked, images = {"yt": {}, "ig": {}}, []
    for corpus in ("yt", "ig"):
        picked[corpus] = {s: 0 for s in QUOTA[corpus]}
    for e in bench:
        corpus = "yt" if e["corpus"] == "ytsheet" else (
            "ig" if e["corpus"] == "igimg" else None)
        if corpus is None:
            continue
        s = stratum(e)
        if picked[corpus].get(s, 99) >= QUOTA[corpus].get(s, 0):
            continue
        if corpus == "yt":
            parsed = yt_second(e["id"])
            if not parsed:
                continue
            base, second = parsed
            new_id = f"yt__{base}_{second}s.jpg"
            out = os.path.join(OUT_IMG, new_id)
            if not os.path.exists(out) and not extract_yt(base, second, out):
                continue
            source = {"type": "yt", "video": base, "second": second,
                      "bench_id": e["id"]}
        else:
            # NOTE: the brief's Interfaces section says IG statics live at
            # data/qwen_scan/frames/igimg__*.jpg, but no such files exist in
            # the current Delay repo (0 matches). bench_set.json's own
            # "path" field is the actual, verified-to-exist location
            # (currently data/ig/media/img/<source>.jpg) for every igimg
            # entry, so we use it directly instead of reconstructing a
            # dead path from the id.
            src_frame = e["path"]
            if not os.path.exists(src_frame):
                continue
            with Image.open(src_frame) as im:
                if im.height < 1080:
                    print("skip (too small):", e["id"], im.size)
                    continue
            new_id = e["id"].replace("igimg__", "ig__")
            out = os.path.join(OUT_IMG, new_id)
            if not os.path.exists(out):
                shutil.copy(src_frame, out)
            source = {"type": "ig", "bench_id": e["id"]}
        with Image.open(out) as im:
            native = [im.width, im.height]
        images.append({"id": new_id, "native": native, "stratum": s, "source": source})
        picked[corpus][s] += 1
        if sum(sum(v.values()) for v in picked.values()) == 40:
            break
    manifest = {"images": sorted(images, key=lambda x: x["id"])}
    json.dump(manifest, open(os.path.join(ROOT, "data", "manifest.json"), "w"), indent=1)
    print(f"{len(images)} images", {c: picked[c] for c in picked})
    if len(images) < 40:
        print("WARNING: quota not filled; consider fetching more videos "
              "(see qwen_res_ladder.py fetch stage in the Delay repo)")


if __name__ == "__main__":
    main()
