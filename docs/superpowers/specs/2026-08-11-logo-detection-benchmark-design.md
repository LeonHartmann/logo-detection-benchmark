# Logo Detection Benchmark: Design

Date: 2026-08-11
Status: approved by Leon (conversation, 2026-08-11)

## Goal

A standalone, reusable benchmark that compares vision LLMs on sponsor-logo
analysis in social media screenshots. It measures three things per model:

1. **Detection**: does the model find each brand's logos (presence), and how
   well does it describe them (size, foreground/background placement)?
2. **Boxes**: how accurate are the bounding boxes it draws (IoU vs human truth)?
3. **Resolution robustness**: how do 1 and 2 degrade as the same screenshot is
   fed in at 1080p, 720p, 480p, 240p, 144p?

First dataset: 40 frames from the Delay Sports corpus (adidas, stripes, dkh,
11teamsports, delay). The harness is brand-agnostic and image-agnostic so
others can point it at their own data.

## Non-goals (v1)

- No per-model prompt tuning: one canonical prompt for every model.
- No video input, no fine-tuning, no local model serving.
- No automatic truth: ground truth is always human-labeled in the bundled UI.

## Repo layout

Location: `/Users/leon/Coding/Research/logo-detection-benchmark`, public-ready.

```
logo-detection-benchmark/
├── README.md                  # quickstart for external users
├── .env.example               # QWEN_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY
├── .gitignore                 # .env, data/images/, data/refs/, results/raw/, derived rungs
├── requirements.txt           # requests, Pillow, PyYAML (keep it to 3)
├── configs/
│   ├── models.yaml            # model rows: name, provider, model id, pricing, enabled
│   └── benchmark.yaml         # rungs, concurrency, retries, jpeg quality, paths
├── brands/
│   └── delay.yaml             # brand list: name, aliases, description, optional ref images
├── data/
│   ├── images/                # GITIGNORED originals (native/1080p res)
│   ├── rungs/                 # GITIGNORED derived downscales
│   ├── manifest.json          # committed: file names, source, native WxH, stratum
│   └── labels/                # committed: one JSON per image, human truth boxes
├── ui/
│   ├── label.html             # draw truth boxes, tag brand/size/placement/location
│   └── review.html            # model-vs-truth disagreement gallery with verdict buttons
├── bench/
│   ├── __init__.py, cli.py    # python -m bench <ladder|run|score|report|serve>
│   ├── providers.py           # OpenAICompatible (OpenAI, DashScope, OpenRouter) + Anthropic
│   ├── prompt.py              # canonical prompt builder incl. optional brand ref images
│   ├── resize.py              # resolution ladder derivation
│   ├── run.py                 # model x image x rung fan-out, resumable JSONL
│   ├── score.py               # presence F1, IoU matching, attributes, robustness, ops
│   └── report.py              # leaderboard.html generator
├── server.py                  # stdlib localhost server: serves ui/, POST /save endpoints
└── results/
    ├── raw/                   # GITIGNORED raw JSONL per model
    ├── scores.json            # committed
    └── leaderboard.html       # committed
```

## Dataset

40 frames, stratified like the existing Delay bench (busy >= 4 logos, normal,
empty for false-positive control, small-logo-heavy):

- ~24 YouTube frames extracted at 1080p from source videos (reuse the videos
  `qwen_res_ladder.py` already downloaded under
  `delay-social-review/data/qwen_res/vid/` where possible; fetch the rest with
  yt-dlp at <=1080p).
- ~16 Instagram statics at native resolution (~1400px wide).

`data/manifest.json` records: id, filename, source (yt video id + timestamp or
ig post id), native size, stratum. Images themselves are gitignored; the
manifest and labels are committed so the benchmark definition is public even
though the images are not redistributed.

## Ground truth: fresh labeling round

Leon labels all 40 frames in `ui/label.html` (served by `server.py`):

- Drag to draw a box; per box choose brand, size (small / medium / large),
  placement (foreground / background), location (chest, sleeve, shorts,
  headwear, board, backdrop, other).
- Size guidance shown in the UI: small = you must squint, large = dominant
  element of the frame.
- Keyboard shortcuts, autosave via POST /save, resumable; frames can be marked
  "no logos" (empty label file is valid truth).
- Label coordinates are stored normalized 0-1000 over the native image.

Estimated effort: ~40 minutes.

## Resolution ladder

Each original is downscaled so image height = rung for rungs
**1080, 720, 480, 240, 144** (configurable in `benchmark.yaml`).

- Rungs larger than the native height are skipped (no upscaling).
- JPEG quality fixed at 85 everywhere so resolution is the only variable.
- Lanczos resampling.
- Models return boxes normalized 0-1000, so the single native-resolution truth
  scores every rung without coordinate translation.

## Models (12, all verified live on the available keys on 2026-08-11)

| # | models.yaml name | Provider (adapter) | Model id | Rationale |
|---|---|---|---|---|
| 1 | qwen3.8-max | DashScope intl (openai-compat) | qwen3.8-max | current production baseline |
| 2 | qwen3-vl-plus | DashScope intl (openai-compat) | qwen3-vl-plus | grounding-trained VL specialist |
| 3 | gpt-5.6-sol | OpenAI (openai-compat) | gpt-5.6-sol | OpenAI frontier tier |
| 4 | gpt-5.6-terra | OpenAI (openai-compat) | gpt-5.6-terra | OpenAI mid tier |
| 5 | gpt-5.6-luna | OpenAI (openai-compat) | gpt-5.6-luna | OpenAI cheap tier; tests the "OpenAI dies at low res" hypothesis per tier |
| 6 | claude-opus-5 | Anthropic | claude-opus-5 | Anthropic frontier |
| 7 | claude-sonnet-5 | Anthropic | claude-sonnet-5 | lineage of the scan2 verifier |
| 8 | gemini-3.6-flash | OpenRouter (openai-compat) | google/gemini-3.6-flash | newest Gemini |
| 9 | gemini-3.1-pro | OpenRouter (openai-compat) | google/gemini-3.1-pro-preview | Gemini pro tier |
| 10 | grok-4.5 | OpenRouter (openai-compat) | x-ai/grok-4.5 | xAI frontier |
| 11 | kimi-k3 | OpenRouter (openai-compat) | moonshotai/kimi-k3 | strong new multimodal |
| 12 | qwen3-vl-235b | OpenRouter (openai-compat) | qwen/qwen3-vl-235b-a22b-instruct | open-weight grounding model, self-hostable if it wins |

Each row carries pricing (USD per 1M input/output tokens) for cost reporting
and an `enabled: true/false` flag. Adding a model is one YAML row.

## Task and prompt contract

One API call per (model, image, rung). The canonical prompt:

- States the brand list with per-brand descriptions/aliases from `brands/*.yaml`.
- If a brand config provides reference images, they are prepended as clearly
  separated "REFERENCE, not the target" images, identically for every model
  (this is the crest-confusion fix from the Qwen tests).
- Asks for every visible logo instance of the listed brands and ONLY compact
  JSON: `{"detections": [{"brand", "box": [x0,y0,x1,y1] ints 0-1000 over the
  target image, "size": "small|medium|large", "placement":
  "foreground|background", "location": "chest|sleeve|shorts|headwear|board|backdrop|other",
  "conf": 1-3}]}`; empty array when nothing is found.

Adapter behavior: temperature 0 where supported, max_tokens capped, one
re-ask retry on unparseable JSON (the retry is recorded; the parse-failure
metric counts first-attempt failures).

## Scoring

Per model per rung, against the 40-frame truth:

- **Presence (headline)**: per brand, frame-level detected-vs-truth ->
  precision / recall / F1; macro-averaged F1 across brands.
- **Boxes**: per brand per frame, sort detections by conf desc, greedy-match to
  unmatched truth boxes by IoU; a pair only matches at IoU >= 0.3. Report hit
  rate at IoU >= 0.5 and >= 0.3, mean IoU of matched pairs. Unmatched detections are FPs (duplicates count as FPs),
  unmatched truth boxes are FNs.
- **Attributes**: on matched pairs (IoU >= 0.3), accuracy of size and placement
  vs truth labels.
- **Resolution robustness**: retention curve = metric at rung / same model's
  metric at that image's highest available rung (1080p for YT frames, native
  height bucket for IG statics), for presence F1 and hit@0.3.
- **Ops**: median and p95 latency, cost per frame from actual usage tokens x
  models.yaml pricing, parse-failure rate.

`scores.json` holds the full breakdown; nothing is collapsed into a single
opaque score.

## Report and human check

`python -m bench report` writes `results/leaderboard.html`:

- Sortable leaderboard: per model x rung, presence F1, hit@0.3/0.5, mean IoU,
  attribute accuracy, cost/frame, latency, parse failures.
- Per-model degradation curves (F1 vs rung).
- Per-brand table (who finds 11teamsports, who hallucinates delay).
- Disagreement gallery: every frame where a model and truth disagree, with
  truth boxes (green) and model boxes (red) overlaid, honest annotated
  examples, no cherry-picking.
- `ui/review.html` mode adds verdict buttons (model right / truth wrong /
  both wrong) whose output is saved and can correct labels for a re-score.

## Run mechanics

- `python -m bench run`: fan-out over enabled models x 40 images x rungs.
  Resumable: results/raw/<model>.jsonl keyed by (image, rung); completed keys
  are skipped on re-run.
- Per-provider concurrency caps (configurable; default 8 DashScope, 8 OpenAI,
  8 Anthropic, 8 OpenRouter), exponential backoff on 429/5xx, per-call timeout.
- Volume: 12 models x 40 images x ~4.6 avg rungs (IG statics skip 1080 when
  native is smaller) ~= 2,200 calls. Estimated cost $15-30 total, wall clock
  30-60 minutes.

## Risks and mitigations

- **A model cannot follow the box format**: parse-failure rate is itself a
  metric; one retry, then the call scores as empty detections.
- **IG statics have no true 1080p rung**: rungs above native height are
  skipped; retention curves are computed against each image's highest
  available rung.
- **Truth errors**: review verdicts feed corrections back into labels; scores
  can be regenerated.
- **Provider-side image downscaling** (e.g. OpenAI detail settings) could mask
  rung differences: adapters request the highest-fidelity image handling the
  API offers (OpenAI `detail: high`) so the input file resolution stays the
  binding constraint.

## Deliverable when done

Leaderboard answering: which model(s) should replace or complement qwen3.8-max
in the Delay pipeline, at which screenshot resolution each model stops being
trustworthy, and what each choice costs per 1,000 frames.
