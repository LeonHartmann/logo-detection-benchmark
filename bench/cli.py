"""python -m bench <ladder|run|score|report|serve|apply-reviews>"""
import argparse
import glob
import json
import os
import sys

from bench import config
from bench.resize import derive


def load_manifest(root):
    return json.load(open(os.path.join(root, "data", "manifest.json")))


def load_labels(root):
    labels = {}
    for p in sorted(glob.glob(os.path.join(root, "data", "labels", "*.json"))):
        d = json.load(open(p))
        if d.get("done"):
            labels[d["image"]] = d["boxes"]
    return labels


def load_raw(root):
    """Load results/raw/*.jsonl, deduped to the last row per (image, rung).

    A retried (image, rung) pair (see bench.run._load_done) can leave more
    than one row for the same key in the file — an old failed attempt
    followed by a later successful one. Scoring should only see the most
    recent attempt, so keep the last row per key in file order.
    """
    raw = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "raw", "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        by_key = {}
        for l in open(p):
            try:
                r = json.loads(l)
                by_key[(r["image"], r["rung"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
        raw[name] = list(by_key.values())
    return raw


def build_parser():
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ladder")
    run_p = sub.add_parser("run")
    run_p.add_argument("--models", default=None, help="comma-separated model names")
    run_p.add_argument("--rungs", default=None, help="comma-separated rungs, e.g. 480,144")
    run_p.add_argument("--limit", type=int, default=None, help="first N images only")
    sub.add_parser("score")
    sub.add_parser("report")
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=8765)
    sub.add_parser("apply-reviews")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = config.ROOT
    config.load_env()
    bench = config.load_bench()
    if args.cmd == "ladder":
        manifest = load_manifest(root)
        for img in manifest["images"]:
            derive(os.path.join(root, "data", "images", img["id"]), img["id"],
                   os.path.join(root, "data", "rungs"), bench["rungs"],
                   bench["jpeg_quality"])
        json.dump([b["name"] for b in config.load_brands()],
                  open(os.path.join(root, "data", "brands.json"), "w"))
        print(f"ladder derived for {len(manifest['images'])} images")
    elif args.cmd == "run":
        from bench.run import run_benchmark
        stats = run_benchmark(
            config.load_models(), bench, config.load_brands(), load_manifest(root),
            root, only_models=args.models.split(",") if args.models else None,
            only_rungs=[int(r) for r in args.rungs.split(",")] if args.rungs else None,
            limit_images=args.limit)
        print(stats)
    elif args.cmd == "score":
        from bench.score import score_all
        raw = load_raw(root)
        labels = load_labels(root)
        # Images that show up in raw model output but have no completed
        # ("done") truth label score as if they had zero truth boxes --
        # silently, since score_all just treats a missing key as `[]`. Warn
        # so a partially-labeled dataset doesn't look like a clean score.
        raw_images = {r["image"] for rows in raw.values() for r in rows}
        missing = sorted(raw_images - set(labels))
        if missing:
            print(f"WARNING: {len(missing)} images have no completed truth "
                  f"labels and score as empty: {', '.join(missing)}")
        scores = score_all(raw, labels, config.load_models(), bench["rungs"])
        out = os.path.join(root, "results", "scores.json")
        json.dump(scores, open(out, "w"), indent=1)
        print("wrote", out)
    elif args.cmd == "report":
        from bench.report import build_report
        build_report(root)
    elif args.cmd == "serve":
        os.execv(sys.executable, [sys.executable,
                                  os.path.join(root, "server.py"), str(args.port)])
    elif args.cmd == "apply-reviews":
        from bench.report import apply_reviews
        apply_reviews(root)


if __name__ == "__main__":
    main()
