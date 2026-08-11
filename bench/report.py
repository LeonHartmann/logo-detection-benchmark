"""Leaderboard HTML, disagreement gallery, and review verdict application."""
import html
import json
import os
import re

from PIL import Image, ImageDraw

from bench.cli import load_labels, load_manifest, load_raw
from bench.score import greedy_match

GALLERY_RUNG = 480  # overlays are rendered on the 480 rung: big enough to see, small files
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def disagreements(dets, truth):
    matches = greedy_match(dets, truth)
    mdi = {di for di, _, _ in matches}
    mti = {ti for _, ti, _ in matches}
    out = []
    for ti, t in enumerate(truth):
        if ti not in mti:
            out.append({"kind": "truth_missed", "brand": t["brand"], "box": t["box"]})
    for di, d in enumerate(dets):
        if di not in mdi:
            out.append({"kind": "model_extra", "brand": d["brand"], "box": d["box"]})
    return out


def render_overlay(image_path, truth_boxes, det_boxes, out_path):
    im = Image.open(image_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for box, color in [(b, "#22c55e") for b in truth_boxes] + \
                      [(b, "#ef4444") for b in det_boxes]:
        x0, y0, x1, y1 = (box[0] * im.width // 1000, box[1] * im.height // 1000,
                          box[2] * im.width // 1000, box[3] * im.height // 1000)
        dr.rectangle([x0, y0, x1, y1], outline=color, width=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, "JPEG", quality=88)


def apply_reviews(root):
    rp = os.path.join(root, "data", "reviews.json")
    if not os.path.exists(rp):
        print("no reviews.json")
        return
    added = removed = 0
    applied_reviews = []
    for r in json.load(open(rp)):
        e, verdict = r.get("entry", {}), r.get("verdict")
        image_id = e.get("image", "")
        if not SAFE_ID.match(image_id):
            continue
        lp = os.path.join(root, "data", "labels", image_id + ".json")
        if not os.path.exists(lp):
            continue
        lab = json.load(open(lp))
        if verdict == "truth_wrong" and e.get("kind") == "truth_missed":
            n0 = len(lab["boxes"])
            lab["boxes"] = [b for b in lab["boxes"]
                            if not (b["brand"] == e["brand"] and b["box"] == e["box"])]
            removed += n0 - len(lab["boxes"])
        elif verdict == "model_right" and e.get("kind") == "model_extra":
            new_box = {"brand": e["brand"], "box": e["box"], "size": "small",
                       "placement": "foreground", "location": "other",
                       "from_review": True}
            if not any(b["brand"] == e["brand"] and b["box"] == e["box"]
                      for b in lab["boxes"]):
                lab["boxes"].append(new_box)
                added += 1
        else:
            continue
        applied_reviews.append(r)
        json.dump(lab, open(lp, "w"), indent=1)
    if applied_reviews:
        app_path = os.path.join(root, "data", "reviews.applied.json")
        existing = []
        if os.path.exists(app_path):
            existing = json.load(open(app_path))
        existing.extend(applied_reviews)
        json.dump(existing, open(app_path, "w"), indent=1)
        os.remove(rp)
    print(f"applied reviews: +{added} boxes, -{removed} boxes; archived to reviews.applied.json; re-run: python -m bench score")


def build_report(root):
    scores = json.load(open(os.path.join(root, "results", "scores.json")))
    raw = load_raw(root)
    labels = load_labels(root)
    gallery_dir = os.path.join(root, "results", "gallery")
    entries = []
    for model, rows in raw.items():
        for row in rows:
            if row["rung"] != GALLERY_RUNG:
                continue
            truth = labels.get(row["image"], [])
            dets = row["detections"] or []
            ds = disagreements(dets, truth)
            if not ds:
                continue
            img_rel = f"gallery/{model}__{row['image']}"
            render_overlay(os.path.join(root, "data", "rungs",
                                        str(GALLERY_RUNG), row["image"]),
                           [t["box"] for t in truth], [d["box"] for d in dets],
                           os.path.join(root, "results", img_rel))
            for i, d in enumerate(ds):
                entries.append({"entry_id": f"{model}|{row['image']}|{i}",
                                "img": img_rel, "model": model,
                                "image": row["image"], "rung": GALLERY_RUNG, **d})
    os.makedirs(gallery_dir, exist_ok=True)
    json.dump(entries, open(os.path.join(gallery_dir, "manifest.json"), "w"), indent=1)
    html_out = _render_html(scores, entries)
    out = os.path.join(root, "results", "leaderboard.html")
    open(out, "w").write(html_out)
    print(f"wrote {out} and {len(entries)} gallery entries")


def _fmt(v):
    """Format metric value: None becomes '-', otherwise use value as-is."""
    return "-" if v is None else v


def _svg_curve(rung_scores):
    rungs = sorted((int(r) for r in rung_scores), reverse=True)
    if len(rungs) < 2:
        return ""
    pts = []
    for i, rg in enumerate(rungs):
        f1 = rung_scores[str(rg)]["presence"]["_macro_f1"] or 0
        pts.append(f"{20 + i * (260 / (len(rungs) - 1)):.0f},{110 - f1 * 100:.0f}")
    labels_x = " ".join(
        f'<text x="{20 + i * (260 / (len(rungs) - 1)):.0f}" y="124" '
        f'font-size="9" text-anchor="middle" fill="#888">{rg}</text>'
        for i, rg in enumerate(rungs))
    return (f'<svg width="300" height="130" viewBox="0 0 300 130">'
            f'<line x1="20" y1="10" x2="20" y2="110" stroke="#444"/>'
            f'<line x1="20" y1="110" x2="280" y2="110" stroke="#444"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="#2563eb" stroke-width="2"/>{labels_x}</svg>')


def _render_html(scores, entries):
    rows = []
    for model, s in sorted(scores["models"].items()):
        for rung, rs in s["rungs"].items():
            o, b, a = rs["ops"], rs["boxes"], rs["attrs"]
            per_brand = ", ".join(f'{k} {v["f1"]}' for k, v in rs["presence"].items()
                                  if not k.startswith("_"))
            rows.append(
                f"<tr><td>{html.escape(model)}</td><td>{rung}</td>"
                f"<td>{_fmt(rs['presence']['_macro_f1'])}</td>"
                f"<td title='{html.escape(per_brand)}'>{_fmt(b['hit03'])}</td>"
                f"<td>{_fmt(b['hit05'])}</td><td>{_fmt(b['mean_iou'])}</td>"
                f"<td>{_fmt(a['size_acc'])}</td><td>{_fmt(a['placement_acc'])}</td>"
                f"<td>{_fmt(o['cost_per_frame'])}</td><td>{_fmt(o['lat_p50'])}</td>"
                f"<td>{_fmt(o['parse_fail_rate'])}</td></tr>")
    curves = "".join(
        f'<div class="curve"><h3>{html.escape(m)}</h3>{_svg_curve(s["rungs"])}</div>'
        for m, s in sorted(scores["models"].items()))
    details_sections = []
    for model, s in sorted(scores["models"].items()):
        detail_rows = []
        for rung, rs in s["rungs"].items():
            for brand, v in sorted(rs["presence"].items()):
                if not brand.startswith("_"):
                    detail_rows.append(
                        f"<tr><td>{html.escape(brand)}</td><td>{_fmt(v.get('p'))}</td>"
                        f"<td>{_fmt(v.get('r'))}</td><td>{_fmt(v.get('f1'))}</td></tr>")
        if detail_rows:
            details_sections.append(
                f'<details><summary>{html.escape(model)} per-brand F1</summary>'
                f'<table style="margin:8px 0;border-collapse:collapse"><thead><tr>'
                f'<th style="border:1px solid #333;padding:4px 8px;text-align:left">brand</th>'
                f'<th style="border:1px solid #333;padding:4px 8px">precision</th>'
                f'<th style="border:1px solid #333;padding:4px 8px">recall</th>'
                f'<th style="border:1px solid #333;padding:4px 8px">f1</th></tr></thead>'
                f'<tbody>{"".join(detail_rows)}</tbody></table></details>')
    gallery = "".join(
        f'<figure><img src="{html.escape(e["img"])}" loading="lazy">'
        f'<figcaption>{html.escape(e["model"])}: {e["kind"]} '
        f'({html.escape(e["brand"])}) on {html.escape(e["image"])}</figcaption></figure>'
        for e in entries)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Logo detection leaderboard</title><style>
body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #333;
padding:4px 8px;text-align:right}}th{{cursor:pointer;background:#1b1b1b}}
td:first-child,th:first-child{{text-align:left}}
.curves{{display:flex;flex-wrap:wrap;gap:16px}}.curve h3{{margin:4px 0;font-size:13px}}
figure{{display:inline-block;margin:8px;max-width:420px}}img{{max-width:100%}}
figcaption{{font-size:12px;color:#aaa}}
</style></head><body>
<h1>Logo detection leaderboard</h1>
<p>Truth boxes are green, model boxes are red in the gallery. Hover the hit@0.3
column for per-brand presence F1. Open ui/review.html via server.py to record
verdicts on disagreements.</p>
<table id="lb"><thead><tr><th>model</th><th>rung</th><th>presence F1</th>
<th>hit@0.3</th><th>hit@0.5</th><th>mean IoU</th><th>size acc</th>
<th>placement acc</th><th>$/frame</th><th>lat p50</th><th>parse fail</th></tr>
</thead><tbody>{"".join(rows)}</tbody></table>
<h2>Per-brand F1 by model</h2>{"".join(details_sections)}
<h2>Presence F1 by resolution</h2><div class="curves">{curves}</div>
<h2>Disagreement gallery ({len(entries)} entries)</h2>{gallery}
<script>
document.querySelectorAll('#lb th').forEach((th,i)=>th.onclick=()=>{{
const tb=document.querySelector('#lb tbody');
[...tb.rows].sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;
const nx=parseFloat(x),ny=parseFloat(y);
return isNaN(nx)||isNaN(ny)?x.localeCompare(y):ny-nx;}})
.forEach(r=>tb.appendChild(r));}});
</script></body></html>"""
