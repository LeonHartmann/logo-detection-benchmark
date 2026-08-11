"""Scoring: presence F1 per brand, greedy IoU box matching, attribute accuracy,
resolution retention, and ops metrics. Pure functions, no I/O except helpers."""
import datetime
import statistics


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def greedy_match(dets, truths, min_iou=0.3):
    """Match detections to truth boxes of the SAME brand, best conf first."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("conf", 2))
    used, matches = set(), []
    for di in order:
        best_ti, best = None, min_iou
        for ti, t in enumerate(truths):
            if ti in used or t["brand"] != dets[di]["brand"]:
                continue
            v = iou(dets[di]["box"], t["box"])
            if v >= best:
                best_ti, best = ti, v
        if best_ti is not None:
            used.add(best_ti)
            matches.append((di, best_ti, best))
    return sorted(matches)


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _score_rung(rows, labels, brands, mcfg):
    pres = {b: {"tp": 0, "fp": 0, "fn": 0} for b in brands}
    hit03 = hit05 = n_truth = n_det = 0
    ious, size_ok, plc_ok, n_matched = [], 0, 0, 0
    lats, cost, fails = [], 0.0, 0
    for row in rows:
        truth = labels.get(row["image"], [])
        dets = row["detections"] if row["parse_ok"] else []
        if not row["parse_ok"]:
            fails += 1
        lats.append(row["latency_s"])
        cost += (row["input_tokens"] * mcfg.price_in
                 + row["output_tokens"] * mcfg.price_out) / 1e6
        t_brands = {t["brand"] for t in truth}
        d_brands = {d["brand"] for d in dets}
        for b in brands:
            if b in t_brands and b in d_brands:
                pres[b]["tp"] += 1
            elif b in d_brands:
                pres[b]["fp"] += 1
            elif b in t_brands:
                pres[b]["fn"] += 1
        n_truth += len(truth)
        n_det += len(dets)
        # Greedy match at min_iou=0.3 for hit03, mean_iou, and attrs
        matches03 = greedy_match(dets, truth)
        for di, ti, v in matches03:
            n_matched += 1
            ious.append(v)
            hit03 += 1
            size_ok += dets[di].get("size") == truth[ti].get("size")
            plc_ok += dets[di].get("placement") == truth[ti].get("placement")
        # Separate greedy match at min_iou=0.5 for hit05 (FINDING 2)
        matches05 = greedy_match(dets, truth, min_iou=0.5)
        hit05 += len(matches05)
    presence = {}
    macro = []
    for b in brands:
        p, r, f1 = _f1(pres[b]["tp"], pres[b]["fp"], pres[b]["fn"])
        presence[b] = {"p": p, "r": r, "f1": f1, **pres[b]}
        if pres[b]["tp"] + pres[b]["fn"] > 0 or pres[b]["fp"] > 0:
            macro.append(f1)
    presence["_macro_f1"] = round(sum(macro) / len(macro), 4) if macro else 0.0
    return {
        "presence": presence,
        "boxes": {"hit03": round(hit03 / n_truth, 4) if n_truth else None,
                  "hit05": round(hit05 / n_truth, 4) if n_truth else None,
                  "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
                  "n_truth": n_truth, "n_det": n_det},
        "attrs": {"size_acc": round(size_ok / n_matched, 4) if n_matched else None,
                  "placement_acc": round(plc_ok / n_matched, 4) if n_matched else None,
                  "n_matched": n_matched},
        "ops": {"lat_p50": round(statistics.median(lats), 2) if lats else None,
                "lat_p95": round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)], 2) if lats else None,
                "cost_per_frame": round(cost / len(rows), 6) if rows else None,
                "parse_fail_rate": round(fails / len(rows), 4) if rows else None,
                "n_frames": len(rows)},
    }


def score_all(raw_by_model, labels, models, rungs):
    brands = sorted({t["brand"] for boxes in labels.values() for t in boxes}
                    | {d["brand"] for rows in raw_by_model.values() for r in rows
                       for d in (r["detections"] or [])})
    out = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "n_images": len(labels), "models": {}}
    mcfgs = {m.name: m for m in models}
    for name, rows in raw_by_model.items():
        by_rung = {}
        for r in rows:
            by_rung.setdefault(r["rung"], []).append(r)
        rung_scores = {str(rg): _score_rung(by_rung[rg], labels, brands, mcfgs[name])
                       for rg in sorted(by_rung, reverse=True)}
        # Use highest configured rung if present, else fallback to max available.
        # Fallback needed because tiny images may produce native-height rungs outside the ladder. (FINDING 3)
        top = str(max(rungs)) if max(rungs) in by_rung else str(max(by_rung))
        retention = {"presence_f1": {}, "hit03": {}}
        for rg in sorted(by_rung, reverse=True):
            if str(rg) == top:
                continue
            for metric, path in (("presence_f1", ("presence", "_macro_f1")),
                                 ("hit03", ("boxes", "hit03"))):
                base = rung_scores[top][path[0]][path[1]]
                cur = rung_scores[str(rg)][path[0]][path[1]]
                # Guard both operands to handle None numerator case. (FINDING 1)
                retention[metric][str(rg)] = (round(cur / base, 4)
                                              if (base and cur is not None) else None)
        out["models"][name] = {"rungs": rung_scores, "retention": retention}
    return out
