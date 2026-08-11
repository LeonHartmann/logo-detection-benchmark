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
        # raw rows written before corner normalization may hold inverted boxes
        dr.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                     outline=color, width=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, "JPEG", quality=88)


def apply_reviews(root):
    rp = os.path.join(root, "data", "reviews.json")
    if not os.path.exists(rp):
        print("no reviews.json")
        return
    added = removed = 0
    all_reviews_with_status = []
    all_reviews = json.load(open(rp))
    for r in all_reviews:
        e, verdict = r.get("entry", {}), r.get("verdict")
        image_id = e.get("image", "")
        was_applied = False
        if SAFE_ID.match(image_id):
            lp = os.path.join(root, "data", "labels", image_id + ".json")
            if os.path.exists(lp):
                lab = json.load(open(lp))
                if verdict == "truth_wrong" and e.get("kind") == "truth_missed":
                    n0 = len(lab["boxes"])
                    lab["boxes"] = [b for b in lab["boxes"]
                                    if not (b["brand"] == e["brand"] and b["box"] == e["box"])]
                    removed += n0 - len(lab["boxes"])
                    was_applied = True
                elif verdict == "model_right" and e.get("kind") == "model_extra":
                    new_box = {"brand": e["brand"], "box": e["box"], "size": "small",
                               "placement": "foreground", "location": "other",
                               "from_review": True}
                    if not any(b["brand"] == e["brand"] and b["box"] == e["box"]
                              for b in lab["boxes"]):
                        lab["boxes"].append(new_box)
                        added += 1
                    was_applied = True
                if was_applied:
                    json.dump(lab, open(lp, "w"), indent=1)
        all_reviews_with_status.append({**r, "applied": was_applied})
    app_path = os.path.join(root, "data", "reviews.applied.json")
    existing = []
    if os.path.exists(app_path):
        existing = json.load(open(app_path))
    existing.extend(all_reviews_with_status)
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
    findings_path = os.path.join(root, "results", "findings.html")
    findings = open(findings_path).read() if os.path.exists(findings_path) else ""
    html_out = _render_html(scores, entries, findings)
    out = os.path.join(root, "results", "leaderboard.html")
    open(out, "w").write(html_out)
    print(f"wrote {out} and {len(entries)} gallery entries")


def _fmt(v):
    """Format metric value: None becomes '-', otherwise use value as-is."""
    return "-" if v is None else v


# Chart palette, validated for the dark card surface (#1a1a19) with the
# dataviz validator: all six pass lightness band, chroma floor, CVD
# separation, normal-vision floor, and 3:1 contrast. Order is fixed.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
SEQ = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
       "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4"]  # low -> high on dark
INK, INK2, MUTED = "#eeeeee", "#c3c2b7", "#898781"
GRID, SURFACE, DIM = "#2c2c2a", "#1a1a19", "#4a4a46"


def _rung_metric(rung_scores, key):
    """The active metric for one rung: presence macro F1 or box hit@0.3."""
    if key == "hit03":
        return rung_scores["boxes"]["hit03"] or 0.0
    return rung_scores["presence"]["_macro_f1"] or 0.0


def _model_stats(scores, key="f1", rung=None):
    """Per-model summary at one rung (default: each model's top rung),
    sorted by the active metric desc."""
    import math
    out = []
    for name, m in scores["models"].items():
        rungs = {int(k): v for k, v in m["rungs"].items()}
        if rung is not None and int(rung) not in rungs:
            continue
        top = rungs[int(rung)] if rung is not None else rungs[max(rungs)]
        cost1k = (top["ops"]["cost_per_frame"] or 0) * 1000
        scored = [v for b, v in top["presence"].items() if not b.startswith("_")
                  and (v.get("tp", 0) + v.get("fn", 0) > 0 or v.get("fp", 0) > 0)]
        out.append({
            "name": name, "rungs": rungs,
            "f1": top["presence"]["_macro_f1"] or 0.0,
            "miou": top["boxes"]["mean_iou"],
            "hit03": top["boxes"]["hit03"] or 0.0,
            "prec": sum(v["p"] for v in scored) / len(scored) if scored else 0.0,
            "rec": sum(v["r"] for v in scored) / len(scored) if scored else 0.0,
            "lat": top["ops"]["lat_p50"],
            "cost1k": cost1k,
            "logc": math.log10(max(cost1k, 0.01)),
            "spend": sum((r["ops"]["cost_per_frame"] or 0) * r["ops"]["n_frames"]
                         for r in rungs.values()),
            "brands": {b: v for b, v in top["presence"].items()
                       if not b.startswith("_")},
        })
    out.sort(key=lambda d: -d[key])
    return out


def _place_labels(labels, y_min, y_max):
    """Greedy vertical collision resolver for point labels.

    labels: [{x, y, text, anchor}] with y as the desired baseline. Labels whose
    horizontal spans overlap are pushed apart vertically by 13px steps.
    """
    est = lambda t: len(t) * 6.2
    spans = []
    for lb in labels:
        w = est(lb["text"])
        x0 = lb["x"] - w if lb["anchor"] == "end" else lb["x"]
        spans.append((x0, x0 + w))
    order = sorted(range(len(labels)), key=lambda i: labels[i]["y"])
    placed = []
    for i in order:
        y = max(y_min, min(labels[i]["y"], y_max))
        for j, yj in placed:
            if not (spans[i][1] < spans[j][0] or spans[j][1] < spans[i][0]):
                if abs(y - yj) < 13:
                    y = yj + 13
        placed.append((i, y))
        labels[i]["y"] = y
    return labels


def _label_svg(labels):
    return "".join(
        f'<text x="{lb["x"]:.1f}" y="{lb["y"]:.1f}" font-size="11" fill="{INK2}" '
        f'text-anchor="{lb["anchor"]}" data-model="{html.escape(lb.get("model", lb["text"]))}">'
        f'{html.escape(lb["text"])}</text>' for lb in labels)


def _dot(x, y, color, tip, r=5, model=""):
    dm = f' data-model="{html.escape(model)}"' if model else ""
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" '
            f'stroke="{SURFACE}" stroke-width="2" data-tt="{html.escape(tip)}"{dm}/>')


def _axis_text(x, y, s, anchor="middle"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="{anchor}" style="font-variant-numeric:tabular-nums">'
            f'{html.escape(str(s))}</text>')


def _chart_cost_quality(stats, key="f1", metric_label="presence F1"):
    """Scatter: cost per 1,000 frames (log x) vs the active metric; Pareto in orange."""
    import math
    W, H, L, R, T, B = 640, 360, 56, 24, 16, 44
    pts = [s for s in stats if s["cost1k"] > 0]
    x0, x1 = math.log10(0.08), math.log10(150)
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    sy = lambda v: T + (1 - v / 0.9) * (H - T - B)
    frontier, best = set(), 0.0
    for s in sorted(pts, key=lambda d: d["cost1k"]):
        if s[key] > max(best, 0.0):
            frontier.add(s["name"]); best = s[key]
    g = []
    for gv in (0.2, 0.4, 0.6, 0.8):
        g.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{W-R}" y2="{sy(gv):.1f}" stroke="{GRID}"/>')
        g.append(_axis_text(L - 8, sy(gv) + 4, f"{gv:.1f}", "end"))
    for tv, lab in ((0.1, "$0.10"), (1, "$1"), (10, "$10"), (100, "$100")):
        g.append(f'<line x1="{sx(math.log10(tv)):.1f}" y1="{T}" x2="{sx(math.log10(tv)):.1f}" y2="{H-B}" stroke="{GRID}"/>')
        g.append(_axis_text(sx(math.log10(tv)), H - B + 16, lab))
    g.append(_axis_text((L + W - R) / 2, H - 6, "cost per 1,000 frames (log)"))
    front_line = sorted((s for s in pts if s["name"] in frontier), key=lambda d: d["logc"])
    g.append('<polyline points="' + " ".join(f"{sx(s['logc']):.1f},{sy(s[key]):.1f}" for s in front_line)
             + f'" fill="none" stroke="{SERIES[1]}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>')
    labels = []
    for s in sorted(pts, key=lambda d: d["name"] in frontier):
        tip = f"{s['name']}: {metric_label} {s[key]:.3f}, ${s['cost1k']:.2f}/1k frames"
        color = SERIES[1] if s["name"] in frontier else SERIES[0]
        g.append(_dot(sx(s["logc"]), sy(s[key]), color, tip, model=s["name"]))
        if s["name"] in frontier or s["name"] == "qwen3.8-max":
            anchor, dx = ("end", -9) if s["logc"] > x1 - 0.7 else ("start", 9)
            labels.append({"x": sx(s["logc"]) + dx, "y": sy(s[key]) + 4,
                           "text": s["name"], "anchor": anchor, "model": s["name"]})
    g.append(_label_svg(_place_labels(labels, T + 10, H - B - 4)))
    return (f'<svg viewBox="0 0 {W} {H}" style="max-width:720px" role="img" '
            f'aria-label="Cost versus quality scatter">' + "".join(g) + "</svg>")


def _chart_presence_boxes(stats):
    """Scatter: presence F1 (x) vs mean IoU (y). Finds brands vs draws boxes."""
    W, H, L, R, T, B = 640, 360, 56, 24, 16, 44
    pts = [s for s in stats if s["miou"] is not None]
    sx = lambda v: L + v / 0.9 * (W - L - R)
    sy = lambda v: T + (1 - v) * (H - T - B)
    g = []
    for gv in (0.2, 0.4, 0.6, 0.8):
        g.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{W-R}" y2="{sy(gv):.1f}" stroke="{GRID}"/>')
        g.append(_axis_text(L - 8, sy(gv) + 4, f"{gv:.1f}", "end"))
        g.append(f'<line x1="{sx(gv):.1f}" y1="{T}" x2="{sx(gv):.1f}" y2="{H-B}" stroke="{GRID}"/>')
        g.append(_axis_text(sx(gv), H - B + 16, f"{gv:.1f}"))
    g.append(_axis_text((L + W - R) / 2, H - 6, "presence F1 at the top rung"))
    g.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" font-size="11" fill="{MUTED}" text-anchor="middle" '
             f'transform="rotate(-90 14 {(T + H - B) / 2:.0f})">mean IoU of matched boxes</text>')
    labeled = {s["name"] for s in sorted(pts, key=lambda d: -(d["miou"] or 0))[:2]} \
        | {s["name"] for s in sorted(pts, key=lambda d: -d["f1"])[:1]} \
        | {"claude-sonnet-5", "qwen3.8-max"}
    labels = []
    for s in pts:
        tip = f"{s['name']}: F1 {s['f1']:.3f}, mean IoU {s['miou']:.3f}, hit@0.3 {s['hit03'] if s['hit03'] is not None else '-'}"
        g.append(_dot(sx(s["f1"]), sy(s["miou"]), SERIES[0], tip, model=s["name"]))
        if s["name"] in labeled:
            anchor, dx = ("end", -9) if s["f1"] > 0.7 else ("start", 9)
            labels.append({"x": sx(s["f1"]) + dx, "y": sy(s["miou"]) + 4,
                           "text": s["name"], "anchor": anchor, "model": s["name"]})
    g.append(_label_svg(_place_labels(labels, T + 10, H - B - 4)))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Presence versus box quality scatter">'
            + "".join(g) + "</svg>")


def _chart_retention(stats, key="f1", metric_label="F1"):
    """Lines: the active metric across resolution rungs; top 6 colored, rest gray."""
    W, H, L, R, T, B = 1180, 380, 56, 150, 16, 44
    rungs = sorted({rg for s in stats for rg in s["rungs"]}, reverse=True)
    if len(rungs) < 2:
        return "", ""
    sx = lambda i: L + i / (len(rungs) - 1) * (W - L - R)
    sy = lambda v: T + (1 - v / 0.9) * (H - T - B)
    top6 = [s["name"] for s in stats[:6]]
    g = []
    for gv in (0.2, 0.4, 0.6, 0.8):
        g.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{W-R}" y2="{sy(gv):.1f}" stroke="{GRID}"/>')
        g.append(_axis_text(L - 8, sy(gv) + 4, f"{gv:.1f}", "end"))
    for i, rg in enumerate(rungs):
        g.append(_axis_text(sx(i), H - B + 16, f"{rg}p"))
    g.append(_axis_text((L + W - R) / 2, H - 6, "resolution rung (image height)"))
    for s in reversed(stats):  # gray lines first, colored on top
        vals = [(i, _rung_metric(s["rungs"][rg], key))
                for i, rg in enumerate(rungs) if rg in s["rungs"]]
        if len(vals) < 2:
            continue
        line = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in vals)
        dm = html.escape(s["name"])
        if s["name"] in top6:
            c = SERIES[top6.index(s["name"])]
            seg = [f'<polyline points="{line}" fill="none" stroke="{c}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>']
            for i, v in vals:
                seg.append(_dot(sx(i), sy(v), c,
                                f"{s['name']} at {rungs[i]}p: {metric_label} {v:.3f}", r=4))
            if s["name"] == top6[0]:
                seg.append(f'<text x="{W-R+8}" y="{sy(vals[-1][1]) + 4:.1f}" font-size="11" '
                           f'fill="{INK2}">{dm}</text>')
            g.append(f'<g data-model="{dm}">' + "".join(seg) + "</g>")
        else:
            g.append(f'<g data-model="{dm}"><polyline points="{line}" fill="none" stroke="{DIM}" '
                     f'stroke-width="1" opacity="0.8"><title>{dm}</title></polyline></g>')
    legend = "".join(
        f'<span class="chip"><i style="background:{SERIES[i]}"></i>{html.escape(n)}</span>'
        for i, n in enumerate(top6)) + \
        f'<span class="chip"><i style="background:{DIM}"></i>{len(stats) - len(top6)} others</span>'
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Presence F1 across resolution rungs">'
           + "".join(g) + "</svg>")
    return svg, legend


def _chart_brand_heatmap(stats):
    """Heatmap: per-brand presence F1 at the top rung, one row per model."""
    brands = {}
    for s in stats:
        for b, v in s["brands"].items():
            brands.setdefault(b, 0)
            brands[b] = max(brands[b], v.get("tp", 0) + v.get("fn", 0))
    order = sorted(brands, key=lambda b: -brands[b])
    CW, RH, L, T = 118, 24, 170, 40
    W = L + CW * len(order) + 16
    H = T + RH * len(stats) + 8
    g = []
    for j, b in enumerate(order):
        g.append(_axis_text(L + j * CW + CW / 2, T - 20, b))
        g.append(_axis_text(L + j * CW + CW / 2, T - 7, f"n={brands[b]} frames"))
    for i, s in enumerate(stats):
        y = T + i * RH
        g.append(f'<g data-model="{html.escape(s["name"])}">')
        g.append(f'<text x="{L - 8}" y="{y + RH / 2 + 4}" font-size="11" fill="{INK2}" '
                 f'text-anchor="end">{html.escape(s["name"])}</text>')
        for j, b in enumerate(order):
            v = s["brands"].get(b, {}).get("f1")
            x = L + j * CW
            if v is None:
                g.append(f'<rect x="{x}" y="{y}" width="{CW - 2}" height="{RH - 2}" fill="{SURFACE}"/>')
                g.append(_axis_text(x + CW / 2, y + RH / 2 + 4, "-"))
                continue
            idx = min(len(SEQ) - 1, max(0, round(v * (len(SEQ) - 1))))
            ink = "#0b0b0b" if idx >= 7 else "#dfe9f7"
            tip = f"{s['name']} / {b}: F1 {v:.3f}"
            g.append(f'<rect x="{x}" y="{y}" width="{CW - 2}" height="{RH - 2}" '
                     f'fill="{SEQ[idx]}" data-tt="{html.escape(tip)}"/>')
            g.append(f'<text x="{x + CW / 2}" y="{y + RH / 2 + 4}" font-size="11" fill="{ink}" '
                     f'text-anchor="middle" style="font-variant-numeric:tabular-nums" '
                     f'pointer-events="none">{v:.2f}</text>')
        g.append("</g>")
    return (f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
            f'aria-label="Per-brand presence F1 heatmap">' + "".join(g) + "</svg>")


def _bar_path(x, y, w, h, r=4):
    """Horizontal bar: square at the baseline (left), 4px rounded data end."""
    if w <= r:
        return f'M{x},{y} h{max(w,1)} v{h} h-{max(w,1)} z'
    return (f'M{x},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r} z')


def _chart_ranked_bars(stats, key="f1", metric_label="presence F1", top_n=12):
    """Ranked horizontal bars for the active metric, value and cost at the tip."""
    rows = stats[:top_n]
    RH, L, T, RPAD = 26, 170, 10, 150
    W = 720
    H = T + RH * len(rows) + 10
    bw = W - L - RPAD
    g = []
    for gv in (0.2, 0.4, 0.6, 0.8):
        x = L + gv / 0.9 * bw
        g.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - 8}" stroke="{GRID}"/>')
    g.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - 8}" stroke="{DIM}"/>')
    for i, s in enumerate(rows):
        y = T + i * RH
        v = s[key]
        w = v / 0.9 * bw
        g.append(f'<g data-model="{html.escape(s["name"])}">')
        g.append(f'<text x="{L - 8}" y="{y + 17}" font-size="12" fill="{INK2}" '
                 f'text-anchor="end">{html.escape(s["name"])}</text>')
        tip = f"{s['name']}: {s[key]:.3f} {metric_label}, ${s['cost1k']:.2f} per 1k frames"
        g.append(f'<path d="{_bar_path(L, y + 4, w, 18)}" fill="{SERIES[0]}" '
                 f'data-tt="{html.escape(tip)}"/>')
        g.append(f'<text x="{L + w + 8:.1f}" y="{y + 17}" font-size="12" fill="{INK}" '
                 f'style="font-variant-numeric:tabular-nums">{v:.3f}'
                 f'<tspan fill="{MUTED}"> · ${s["cost1k"]:.2f}/1k</tspan></text>')
        g.append('</g>')
    return (f'<svg viewBox="0 0 {W} {H}" style="max-width:760px" role="img" '
            f'aria-label="Ranked {metric_label} bars">' + "".join(g) + "</svg>")


def _chart_precision_recall(stats):
    """Scatter: macro presence precision vs recall at the top rung."""
    W, H, L, R, T, B = 640, 360, 56, 24, 16, 44
    pts = [s for s in stats if s["f1"] > 0.05]
    lo = 0.3
    sx = lambda v: L + (max(v, lo) - lo) / (1 - lo) * (W - L - R)
    sy = lambda v: T + (1 - (max(v, lo) - lo) / (1 - lo)) * (H - T - B)
    g = []
    for gv in (0.4, 0.6, 0.8, 1.0):
        g.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{W-R}" y2="{sy(gv):.1f}" stroke="{GRID}"/>')
        g.append(_axis_text(L - 8, sy(gv) + 4, f"{gv:.1f}", "end"))
        g.append(f'<line x1="{sx(gv):.1f}" y1="{T}" x2="{sx(gv):.1f}" y2="{H-B}" stroke="{GRID}"/>')
        g.append(_axis_text(sx(gv), H - B + 16, f"{gv:.1f}"))
    g.append(f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" '
             f'stroke="{DIM}" stroke-dasharray="3 3" opacity="0.5"/>')
    g.append(_axis_text((L + W - R) / 2, H - 6, "macro recall: share of present brands reported"))
    g.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" font-size="11" fill="{MUTED}" text-anchor="middle" '
             f'transform="rotate(-90 14 {(T + H - B) / 2:.0f})">macro precision: reports that were real</text>')
    by_p = sorted(pts, key=lambda d: -d["prec"])
    by_r = sorted(pts, key=lambda d: -d["rec"])
    labeled = {by_p[0]["name"], by_p[-1]["name"], by_r[0]["name"], "qwen3.8-max"}
    labels = []
    for s in pts:
        tip = (f"{s['name']}: precision {s['prec']:.3f}, recall {s['rec']:.3f}, "
               f"F1 {s['f1']:.3f}")
        g.append(_dot(sx(s["rec"]), sy(s["prec"]), SERIES[0], tip, model=s["name"]))
        if s["name"] in labeled:
            anchor_, dx = ("end", -9) if s["rec"] > 0.85 else ("start", 9)
            labels.append({"x": sx(s["rec"]) + dx, "y": sy(s["prec"]) + 4,
                           "text": s["name"], "anchor": anchor_, "model": s["name"]})
    g.append(_label_svg(_place_labels(labels, T + 10, H - B - 4)))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Precision versus recall scatter">'
            + "".join(g) + "</svg>")


def _chart_latency(stats):
    """Dot plot: median seconds per call at the top rung, log scale."""
    import math
    rows = sorted(stats, key=lambda s: s["lat"] or 9e9)
    RH, L, T = 20, 170, 24
    W, RPAD = 720, 60
    H = T + RH * len(rows) + 12
    x0, x1 = math.log10(1), math.log10(300)
    sx = lambda v: L + (math.log10(max(v, 1)) - x0) / (x1 - x0) * (W - L - RPAD)
    g = []
    for tv in (1, 3, 10, 30, 100, 300):
        g.append(f'<line x1="{sx(tv):.1f}" y1="{T - 6}" x2="{sx(tv):.1f}" y2="{H - 8}" stroke="{GRID}"/>')
        g.append(_axis_text(sx(tv), T - 10, f"{tv}s"))
    for i, s in enumerate(rows):
        if s["lat"] is None:
            continue
        y = T + i * RH + RH / 2
        g.append(f'<g data-model="{html.escape(s["name"])}">')
        g.append(f'<text x="{L - 8}" y="{y + 4}" font-size="11" fill="{INK2}" '
                 f'text-anchor="end">{html.escape(s["name"])}</text>')
        g.append(f'<line x1="{L}" y1="{y:.1f}" x2="{sx(s["lat"]):.1f}" y2="{y:.1f}" '
                 f'stroke="{GRID}"/>')
        g.append(_dot(sx(s["lat"]), y, SERIES[0], f"{s['name']}: median {s['lat']:.1f}s per call", r=4.5))
        g.append(f'<text x="{sx(s["lat"]) + 9:.1f}" y="{y + 4}" font-size="11" fill="{MUTED}" '
                 f'style="font-variant-numeric:tabular-nums">{s["lat"]:.1f}s</text>')
        g.append('</g>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Median latency dot plot">'
            + "".join(g) + "</svg>")


def _col_path(x, y_top, w, h, r=4):
    """Vertical column: 4px rounded data end (top), square at the baseline."""
    if h <= r + 1:
        return f'M{x},{y_top} h{w} v{h} h-{w} z'
    return (f'M{x},{y_top + r} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r} '
            f'a{r},{r} 0 0 1 {r},{r} v{h - r} h-{w} z')


def _method_groups(scores):
    """base model -> {condition: row name} for every <base>+<cond> row."""
    groups = {}
    for name in scores["models"]:
        if "+" in name:
            base, cond = name.split("+", 1)
            if base in scores["models"]:
                groups.setdefault(base, {})[cond] = name
    return groups


def _chart_method_bars(scores, metric, y_max=0.9):
    """Grouped columns comparing base vs +refs vs +zoom per model."""
    groups = _method_groups(scores)
    if not groups:
        return ""
    conds = [("base", None, SERIES[0]), ("refs", "refs", SERIES[1]),
             ("zoom", "zoom", SERIES[2]), ("perbrand", "perbrand", SERIES[3])]
    vals = {}
    for base, variants in groups.items():
        for label, cond, _ in conds:
            name = base if cond is None else variants.get(cond)
            if name:
                v = metric(scores["models"][name])
                if v is not None:
                    vals[(base, label)] = v
    bases = sorted(groups, key=lambda b: -vals.get((b, "base"), 0))
    BW, IGAP, GGAP, L, T, B = 24, 2, 34, 44, 14, 46
    gw = 4 * BW + 3 * IGAP
    W = L + len(bases) * (gw + GGAP) + 10
    H = 300
    sy = lambda v: T + (1 - v / y_max) * (H - T - B)
    g = []
    for gv in (0.2, 0.4, 0.6, 0.8):
        g.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{W - 10}" y2="{sy(gv):.1f}" stroke="{GRID}"/>')
        g.append(_axis_text(L - 8, sy(gv) + 4, f"{gv:.1f}", "end"))
    g.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{W - 10}" y2="{sy(0):.1f}" stroke="{DIM}"/>')
    for gi, base in enumerate(bases):
        gx = L + gi * (gw + GGAP)
        for ci, (label, cond, color) in enumerate(conds):
            v = vals.get((base, label))
            if v is None:
                continue
            x = gx + ci * (BW + IGAP)
            row_name = base if cond is None else groups[base][cond]
            h = sy(0) - sy(v)
            g.append(f'<g data-model="{html.escape(row_name)}">')
            g.append(f'<path d="{_col_path(x, sy(v), BW, h)}" fill="{color}" '
                     f'data-tt="{html.escape(f"{row_name}: {v:.3f}")}"/>')
            g.append(f'<text x="{x + BW / 2:.1f}" y="{sy(v) - 4:.1f}" font-size="9" '
                     f'fill="{INK2}" text-anchor="middle" '
                     f'style="font-variant-numeric:tabular-nums">{v:.2f}</text>')
            g.append('</g>')
        g.append(_axis_text(gx + gw / 2, H - B + 16, base))
    legend = "".join(
        f'<span class="chip"><i style="background:{c}"></i>{lb}</span>'
        for lb, _, c in [("base", None, SERIES[0]), ("with refs", None, SERIES[1]),
                         ("with zoom", None, SERIES[2]),
                         ("per-brand calls + refs", None, SERIES[3])])
    return (f'<div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" style="max-width:{W}px" role="img" '
            f'aria-label="Method comparison columns">' + "".join(g) + "</svg>")


def _stat_tiles(scores, stats, key="f1", metric_label="F1"):
    calls = sum(r["ops"]["n_frames"] for s in stats for r in s["rungs"].values())
    spend = sum(s["spend"] for s in stats)
    best = stats[0]
    value = min((s for s in stats if s[key] >= best[key] - 0.05),
                key=lambda s: s["cost1k"], default=best)
    tiles = [
        ("models", str(len(stats)), "x 5 resolution rungs"),
        ("images", str(scores["n_images"]), "human-labeled truth"),
        ("API calls", f"{calls:,}", "all recorded and resumable"),
        ("total spend", f"${spend:,.2f}", "from real token usage"),
        (f"best {metric_label}", f'{best[key]:.3f}', best["name"]),
        ("value pick", value["name"],
         f'{metric_label} {value[key]:.3f} at ${value["cost1k"]:.2f}/1k'),
    ]
    return "".join(
        f'<div class="tile"><div class="tl">{html.escape(l)}</div>'
        f'<div class="tv">{html.escape(v)}</div>'
        f'<div class="ts">{html.escape(s)}</div></div>' for l, v, s in tiles)


def _render_html(scores, entries, findings=""):
    all_rungs = sorted({int(r) for m in scores["models"].values()
                        for r in m["rungs"]}, reverse=True)
    top_rung = all_rungs[0]

    # ---------- full results table (filterable by model and rung) ----------
    rows = []
    for model, s_ in sorted(scores["models"].items()):
        for rung, rs in s_["rungs"].items():
            o, b, a = rs["ops"], rs["boxes"], rs["attrs"]
            per_brand = ", ".join(f'{k} {_fmt(v["f1"])}' for k, v in rs["presence"].items()
                                  if not k.startswith("_"))
            rows.append(
                f'<tr data-model="{html.escape(model)}" data-rung="{rung}">'
                f"<td>{html.escape(model)}</td><td>{rung}</td>"
                f"<td>{_fmt(rs['presence']['_macro_f1'])}</td>"
                f"<td title='{html.escape(per_brand)}'>{_fmt(b['hit03'])}</td>"
                f"<td>{_fmt(b['hit05'])}</td><td>{_fmt(b['mean_iou'])}</td>"
                f"<td>{_fmt(a['size_acc'])}</td><td>{_fmt(a['placement_acc'])}</td>"
                f"<td>{_fmt(o['cost_per_frame'])}</td><td>{_fmt(o['lat_p50'])}</td>"
                f"<td>{_fmt(o['parse_fail_rate'])}</td></tr>")

    # ---------- per-rung chart bundles ----------
    view_defs = [("f1", "presence F1"), ("hit03", "hit@0.3")]
    bundles = []
    for rung in all_rungs:
        rl = f"{rung}p"
        for key, mlabel in view_defs:
            vstats = _model_stats(scores, key, rung)
            bundles.append(f"""
<div data-bundle data-view="{key}" data-rung="{rung}">
<div class="tiles">{_stat_tiles(scores, vstats, key, mlabel)}</div>
<div class="cards">
<div class="card"><h3>Top models by {mlabel}</h3>
<p class="sub">At {rl}. Value at the bar tip, price per 1,000 frames beside it.</p>
{_chart_ranked_bars(vstats, key, mlabel)}</div>
<div class="card"><h3>Cost vs quality</h3>
<p class="sub">{mlabel} at {rl} against price per 1,000 frames (log scale).
Orange marks the Pareto frontier: nothing cheaper scores higher.</p>
{_chart_cost_quality(vstats, key, mlabel)}</div>
</div>
</div>""")
        rstats = _model_stats(scores, "f1", rung)
        mp = lambda m, r=rung: (m["rungs"].get(str(r)) or {}).get("presence", {}).get("_macro_f1")
        mh = lambda m, r=rung: (m["rungs"].get(str(r)) or {}).get("boxes", {}).get("hit03")
        bundles.append(f"""
<div data-bundle data-view="*" data-rung="{rung}">
<h2>Method comparison at {rl}: plain vs reference images vs zoom vs per-brand</h2>
<div class="cards">
<div class="card"><h3>Finding brands: presence F1</h3>
<p class="sub">Same model, different ways of asking. Missing bars mean the
condition was not run for that model.</p>
{_chart_method_bars(scores, mp)}</div>
<div class="card"><h3>Drawing boxes: hit@0.3</h3>
<p class="sub">Share of truth boxes matched at IoU 0.3 or better.</p>
{_chart_method_bars(scores, mh, y_max=0.7)}</div>
</div>
<h2>Diagnostics at {rl}</h2>
<div class="cards">
<div class="card"><h3>Hallucination check: precision vs recall</h3>
<p class="sub">Below the diagonal means the model reports brands that are not
there; right of it means it finds what exists. Models with F1 above 0.05.</p>
{_chart_precision_recall(rstats)}</div>
<div class="card"><h3>Speed: median seconds per call</h3>
<p class="sub">Median latency at {rl}, log scale. Reasoning models think long;
specialists answer fast.</p>
{_chart_latency(rstats)}</div>
<div class="card"><h3>Finding brands vs drawing boxes</h3>
<p class="sub">Presence F1 against mean IoU of matched boxes. Top right is the
goal; bottom right knows a brand is present but cannot localize it.</p>
{_chart_presence_boxes(rstats)}</div>
<div class="card"><h3>Per-brand presence F1</h3>
<p class="sub">Darker is worse, lighter is better. Columns ordered by truth
frame count; treat low-n columns as anecdotes.</p>
<div style="overflow-x:auto">{_chart_brand_heatmap(rstats)}</div></div>
</div>
</div>""")

    # retention: rung-independent, one per view
    for key, mlabel in view_defs:
        vstats = _model_stats(scores, key)
        ret_svg, ret_leg = _chart_retention(vstats, key, mlabel)
        bundles.append(f"""
<div data-bundle data-view="{key}" data-rung="*">
<div class="card"><h3>Resolution robustness</h3>
<p class="sub">{mlabel} as the same screenshots shrink from {top_rung}p to
{all_rungs[-1]}p. The flatter the line, the more resolution-proof the model.
This chart always shows every rung.</p>
<div class="legend">{ret_leg}</div>
{ret_svg}</div>
</div>""")

    # ---------- header numbers, controls, details, gallery ----------
    stats = _model_stats(scores, "f1")
    n_models = len(stats)
    n_images = scores["n_images"]
    n_calls = sum(r["ops"]["n_frames"] for st in stats for r in st["rungs"].values())
    spend = sum(st["spend"] for st in stats)
    gen_date = (scores.get("generated") or "")[:10]

    model_names = [st["name"] for st in stats]
    baselines = [n for n in model_names if "+" not in n]
    top10 = model_names[:10]
    chips = "".join(f'<button class="fchip on" data-fmodel="{html.escape(n)}">'
                    f'{html.escape(n)}</button>' for n in model_names)
    rung_seg = "".join(
        f'<button class="segbtn rungbtn{" on" if r == top_rung else ""}" '
        f'data-rung="{r}">{r}p</button>' for r in all_rungs)

    details_sections = []
    for model, s_ in sorted(scores["models"].items()):
        detail_rows = []
        for rung, rs in s_["rungs"].items():
            for brand, v in sorted(rs["presence"].items()):
                if not brand.startswith("_"):
                    detail_rows.append(
                        f"<tr><td>{html.escape(brand)}</td><td>{rung}</td>"
                        f"<td>{_fmt(v.get('p'))}</td>"
                        f"<td>{_fmt(v.get('r'))}</td><td>{_fmt(v.get('f1'))}</td></tr>")
        if detail_rows:
            details_sections.append(
                f'<details data-model="{html.escape(model)}">'
                f'<summary>{html.escape(model)} per-brand F1</summary>'
                f'<table class="mini"><thead><tr><th>brand</th><th>rung</th>'
                f'<th>precision</th><th>recall</th><th>f1</th></tr></thead>'
                f'<tbody>{"".join(detail_rows)}</tbody></table></details>')

    gallery_json = json.dumps([
        {"m": e["model"], "img": e["img"], "k": e["kind"],
         "b": e["brand"], "i": e["image"]} for e in entries]).replace("</", "<\\/")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Logo detection benchmark</title><style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
background:#0d0d0d;color:#eee}}
.wrap{{max-width:1300px;margin:0 auto;padding:28px 28px 60px}}
header{{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;flex-wrap:wrap}}
h1{{margin:0 0 6px;font-size:24px;letter-spacing:-.01em}}
h2{{font-size:16px;margin:36px 0 12px;letter-spacing:-.005em}}
.eyebrow{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#898781}}
.meta{{text-align:right;font-size:13px;color:#c3c2b7;padding-top:8px;white-space:nowrap}}
.meta span{{font-size:12px;color:#898781}}
.sub{{color:#898781;font-size:12px;margin:0 0 12px}}
a{{color:#3987e5}}
/* toolbar */
.toolbar{{position:sticky;top:0;z-index:6;background:rgba(13,13,13,.94);
backdrop-filter:blur(6px);border-bottom:1px solid rgba(255,255,255,.07);
margin:18px -28px 20px;padding:10px 28px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}}
.seg{{display:inline-flex;border:1px solid rgba(255,255,255,.14);border-radius:8px;overflow:hidden}}
.segbtn{{font:12px system-ui;background:transparent;color:#c3c2b7;border:none;
padding:6px 12px;cursor:pointer}}
.segbtn:hover{{background:rgba(255,255,255,.06)}}
.segbtn.on{{background:#3987e5;color:#0b0b0b;font-weight:600}}
.tlabel{{font-size:11px;color:#898781;letter-spacing:.06em;text-transform:uppercase;margin:0 2px 0 8px}}
.fbtn{{font-size:12px;color:#c3c2b7;cursor:pointer;background:transparent;
border:1px solid rgba(255,255,255,.14);border-radius:8px;padding:5px 10px}}
.fbtn:hover{{background:rgba(255,255,255,.06)}}
#modelbox{{width:100%;order:9}}
#modelbox summary{{cursor:pointer;font-size:12px;color:#c3c2b7;list-style:none;padding:2px 0}}
#modelbox summary::before{{content:"▸ "}}#modelbox[open] summary::before{{content:"▾ "}}
.chiprow{{display:flex;flex-wrap:wrap;gap:6px;padding:8px 0 2px}}
.fchip{{font:12px system-ui;background:#1a1a19;color:#c3c2b7;
border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:3px 10px;cursor:pointer}}
.fchip:hover{{border-color:rgba(255,255,255,.3)}}
.fchip.on{{background:#24303f;color:#eee;border-color:#3987e5}}
button:focus-visible,.fchip:focus-visible{{outline:2px solid #3987e5;outline-offset:1px}}
/* cards and tiles */
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}}
.tile{{background:#1a1a19;border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:12px 14px}}
.tl{{font-size:11px;color:#898781;letter-spacing:.06em;text-transform:uppercase}}
.tv{{font-size:24px;font-weight:600;margin:2px 0}}
.ts{{font-size:12px;color:#c3c2b7}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:16px;margin:14px 0}}
.card{{background:#1a1a19;border:1px solid rgba(255,255,255,.07);border-radius:10px;
padding:18px;margin:14px 0}}
.cards .card{{margin:0}}
.card h3{{margin:0 0 4px;font-size:15px;letter-spacing:-.005em}}
svg{{max-width:100%;height:auto;display:block}}
svg text{{font-family:system-ui,-apple-system,sans-serif}}
.legend{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px}}
.chip{{font-size:12px;color:#c3c2b7;display:inline-flex;align-items:center;gap:6px}}
.chip i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
/* table */
table{{border-collapse:separate;border-spacing:0;width:100%;background:#1a1a19;
border:1px solid rgba(255,255,255,.07);border-radius:10px;overflow:hidden}}
th,td{{padding:7px 10px;text-align:right;font-variant-numeric:tabular-nums;font-size:13px;
border-bottom:1px solid rgba(255,255,255,.05)}}
th{{cursor:pointer;background:#232322;color:#c3c2b7;font-size:11px;letter-spacing:.05em;
text-transform:uppercase}}
#lb th{{position:sticky;top:53px;z-index:2}}
td:first-child,th:first-child{{text-align:left}}
tbody tr:nth-child(odd) td{{background:rgba(255,255,255,.015)}}
tbody tr:hover td{{background:rgba(57,135,229,.08)}}
.mini{{margin:8px 0;border-radius:8px}}
.mini th{{position:static}}
details[data-model]{{margin:4px 0}}
details[data-model] summary{{cursor:pointer;font-size:13px;color:#c3c2b7;padding:3px 0}}
/* gallery */
#gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
#gal figure{{margin:0;background:#1a1a19;border:1px solid rgba(255,255,255,.07);
border-radius:10px;overflow:hidden}}
#gal img{{width:100%;display:block}}
#gal figcaption{{font-size:12px;color:#c3c2b7;padding:8px 10px}}
#galmore{{margin:16px auto;display:block}}
#tt{{position:fixed;pointer-events:none;background:#0b0b0b;color:#eee;
border:1px solid rgba(255,255,255,.15);border-radius:6px;padding:6px 10px;font-size:12px;
display:none;z-index:10;max-width:320px}}
[data-model].fdim{{opacity:0.12;pointer-events:none}}
tr.fhide,details.fhide{{display:none}}
footer{{margin:48px 0 0;color:#898781;font-size:12px;border-top:1px solid rgba(255,255,255,.07);
padding-top:16px}}
@media (prefers-reduced-motion:no-preference){{.fchip,.segbtn,.fbtn{{transition:background .12s,border-color .12s}}}}
</style></head><body><div class="wrap">
<header>
<div>
<div class="eyebrow">Vision model benchmark · Delay Sports corpus</div>
<h1>Logo detection benchmark</h1>
<p class="sub" style="max-width:740px">{n_models} vision models, {n_images} human-labeled
social media frames, 5 brands, {len(all_rungs)} resolution rungs ({top_rung}p to
{all_rungs[-1]}p). One canonical prompt, boxes scored against human truth at IoU
thresholds. Hover any mark for exact values.</p>
</div>
<div class="meta">Research by Leon Hartmann<br>
<span>generated {gen_date} · {n_calls:,} API calls · ${spend:,.2f} total</span></div>
</header>
{findings}
<div class="toolbar">
<span class="tlabel">Ranking</span>
<span class="seg"><button class="segbtn viewbtn on" data-view="f1">Logo detection only</button><button class="segbtn viewbtn" data-view="hit03">Detection + boxes</button></span>
<span class="tlabel">Resolution</span>
<span class="seg">{rung_seg}</span>
<span class="tlabel">Models</span>
<button class="fbtn" data-preset="all">all</button>
<button class="fbtn" data-preset="none">none</button>
<button class="fbtn" data-preset="base">baselines</button>
<button class="fbtn" data-preset="cond">conditions</button>
<button class="fbtn" data-preset="top10">top 10</button>
<details id="modelbox"><summary>pick individual models</summary>
<div class="chiprow">{chips}</div></details>
</div>
<p class="sub">Logo detection only ranks by presence F1: did the model report the
brand at all. Detection + boxes ranks by hit@0.3: the share of truth logos the
model found AND boxed at IoU 0.3 or better. The resolution buttons rescope every
chart and the table to that rung.</p>
{"".join(bundles)}
<h2>Full results table</h2>
<p class="sub">Showing the selected resolution only. Click a column header to sort.</p>
<table id="lb"><thead><tr><th>model</th><th>rung</th><th>presence F1</th>
<th>hit@0.3</th><th>hit@0.5</th><th>mean IoU</th><th>size acc</th>
<th>placement acc</th><th>$/frame</th><th>lat p50</th><th>parse fail</th></tr>
</thead><tbody>{"".join(rows)}</tbody></table>
<h2>Per-brand F1 by model</h2>{"".join(details_sections)}
<h2 id="galh">Disagreement gallery</h2>
<p class="sub">Rendered at 480p. Truth boxes green, model boxes red. Filtered by
the model selection above.</p>
<div id="gal"></div>
<button id="galmore" class="fbtn">show more</button>
<footer>Logo detection benchmark · Research by Leon Hartmann · generated {gen_date}</footer>
</div>
<div id="tt"></div>
<script id="galdata" type="application/json">{gallery_json}</script>
<script>
const MODELS={json.dumps(model_names)};
const BASELINES=new Set({json.dumps(baselines)});
const TOP10=new Set({json.dumps(top10)});
const state={{view:'f1',rung:'{top_rung}',models:new Set(MODELS)}};
try{{const v=localStorage.getItem('lb_view');if(v)state.view=v;
const r=localStorage.getItem('lb_rung');if(r)state.rung=r;}}catch(e){{}}
const GAL=JSON.parse(document.getElementById('galdata').textContent);
let galShown=0;const GALBATCH=48;
function galRender(reset){{
  const gal=document.getElementById('gal');
  if(reset){{gal.innerHTML='';galShown=0;}}
  const list=GAL.filter(e=>state.models.has(e.m));
  document.getElementById('galh').textContent=
    `Disagreement gallery (${{list.length}} entries)`;
  const next=list.slice(galShown,galShown+GALBATCH);
  for(const e of next){{
    const f=document.createElement('figure');
    f.innerHTML=`<img src="${{e.img}}" loading="lazy"><figcaption></figcaption>`;
    f.querySelector('figcaption').textContent=`${{e.m}}: ${{e.k}} (${{e.b}}) on ${{e.i}}`;
    gal.appendChild(f);
  }}
  galShown+=next.length;
  document.getElementById('galmore').style.display=
    galShown<list.length?'block':'none';
}}
document.getElementById('galmore').onclick=()=>galRender(false);
function apply(){{
  document.querySelectorAll('.viewbtn').forEach(b=>
    b.classList.toggle('on',b.dataset.view===state.view));
  document.querySelectorAll('.rungbtn').forEach(b=>
    b.classList.toggle('on',b.dataset.rung===state.rung));
  document.querySelectorAll('[data-bundle]').forEach(el=>{{
    const vOk=el.dataset.view==='*'||el.dataset.view===state.view;
    const rOk=el.dataset.rung==='*'||el.dataset.rung===state.rung;
    el.style.display=vOk&&rOk?'':'none';
  }});
  document.querySelectorAll('.fchip').forEach(c=>
    c.classList.toggle('on',state.models.has(c.dataset.fmodel)));
  document.querySelectorAll('[data-model]').forEach(el=>{{
    const on=state.models.has(el.dataset.model);
    if(el.tagName==='TR'){{
      el.classList.toggle('fhide',!on||el.dataset.rung!==state.rung);
    }} else if(el.tagName==='DETAILS'){{
      el.classList.toggle('fhide',!on);
    }} else {{
      el.classList.toggle('fdim',!on);
    }}
  }});
  try{{localStorage.setItem('lb_view',state.view);
  localStorage.setItem('lb_rung',state.rung);}}catch(e){{}}
  galRender(true);
}}
document.querySelectorAll('.viewbtn').forEach(b=>b.onclick=()=>{{state.view=b.dataset.view;apply();}});
document.querySelectorAll('.rungbtn').forEach(b=>b.onclick=()=>{{state.rung=b.dataset.rung;apply();}});
document.querySelectorAll('.fchip').forEach(c=>c.onclick=()=>{{
  const m=c.dataset.fmodel;
  state.models.has(m)?state.models.delete(m):state.models.add(m);apply();}});
document.querySelectorAll('.fbtn[data-preset]').forEach(b=>b.onclick=()=>{{
  const p=b.dataset.preset;
  if(p==='all')state.models=new Set(MODELS);
  else if(p==='none')state.models=new Set();
  else if(p==='base')state.models=new Set(MODELS.filter(m=>BASELINES.has(m)));
  else if(p==='cond')state.models=new Set(MODELS.filter(m=>m.includes('+')).flatMap(m=>[m,m.split('+')[0]]));
  else if(p==='top10')state.models=new Set(MODELS.filter(m=>TOP10.has(m)));
  apply();}});
const tt=document.getElementById('tt');
document.querySelectorAll('[data-tt]').forEach(el=>{{
el.addEventListener('mousemove',e=>{{tt.textContent=el.dataset.tt;
tt.style.display='block';tt.style.left=Math.min(e.clientX+14,innerWidth-330)+'px';
tt.style.top=(e.clientY+14)+'px';}});
el.addEventListener('mouseleave',()=>tt.style.display='none');}});
document.querySelectorAll('#lb th').forEach((th,i)=>th.onclick=()=>{{
const tb=document.querySelector('#lb tbody');
[...tb.rows].sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;
const nx=parseFloat(x),ny=parseFloat(y);
return isNaN(nx)||isNaN(ny)?x.localeCompare(y):ny-nx;}})
.forEach(r=>tb.appendChild(r));}});
apply();
</script></body></html>"""
