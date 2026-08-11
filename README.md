# logo-detection-benchmark

A standalone benchmark for comparing vision LLMs on brand-logo detection in
social media screenshots. It measures three things per model: whether it
finds each brand's logo instances (presence), how accurate its bounding
boxes are (box quality), and how much accuracy it loses as the same image is
fed in at lower resolutions (resolution robustness). Results are reported
per model, per resolution rung, against a human-labeled ground truth set.

The harness is brand-agnostic and image-agnostic: the bundled dataset covers
five brands from the Delay Sports corpus, but the config, labeling UI, and
scoring all work against any set of images and brands you supply.

## Quickstart

1. Clone the repo.
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in the API keys for the providers you
   plan to use (only the providers referenced by enabled rows in
   `configs/models.yaml` are needed).
4. Put your images in `data/images/` (this directory is gitignored; images
   are never committed).
5. `python -m bench manifest`: scans `data/images/` and writes
   `data/manifest.json` with every image's id and native pixel size. Run it
   again whenever you add or remove images: it merges by filename, so
   existing entries keep whatever `stratum` and `source` you already set on
   them, new files get `stratum: "unlabeled"` with an empty `source`, and
   entries for files no longer on disk are dropped (each drop is printed).
   Schema, with a two-image example, useful as a reference when you hand-edit
   `stratum` and `source` afterward:

   ```json
   {
    "images": [
     {
      "id": "example_001.jpg",
      "native": [1920, 1080],
      "stratum": "normal",
      "source": {"type": "custom", "note": "frame from my_video, t=12s"}
     },
     {
      "id": "example_002.jpg",
      "native": [1440, 1800],
      "stratum": "small",
      "source": {"type": "custom", "note": "static image, my_source_id"}
     }
    ]
   }
   ```

   `id` matches the filename under `data/images/`. `native` is
   `[width, height]` in pixels, filled in automatically by the command
   above. `stratum` is any short label you find useful for grouping results
   later (for example busy, normal, empty, small-logo); edit it by hand once
   you know what's in each image.
6. Set up your brands: see "Use your own brands" below.
7. `python -m bench ladder`: derives the resolution rungs (from
   `configs/benchmark.yaml`, default 1080/720/480/240/144) for every image in
   the manifest.
8. `python server.py` and label ground truth at
   `http://localhost:8765/ui/label.html`: draw a box per logo instance, tag
   its brand, size, placement, and location, and save. Frames with no logos
   are marked "no logos" and count as valid (empty) truth. Optionally, run
   `python -m bench prelabel` first (see "Pre-labeling (optional)" below) so
   most frames start with a set of suggested boxes to review instead of an
   empty canvas.
9. `python -m bench run`: calls every enabled model in `configs/models.yaml`
   against every image at every rung. Resumable: it skips (image, rung)
   pairs already recorded in `results/raw/<model>.jsonl`.
10. `python -m bench score`: computes all metrics from the raw results and
    the labels, writing `results/scores.json`.
11. `python -m bench report`: builds `results/leaderboard.html` from the
    scores, including the disagreement gallery.
12. Open `results/leaderboard.html` in a browser.

### Pre-labeling (optional)

`python -m bench prelabel [--model NAME] [--rung R]` runs one configured
model over every manifest image and writes its detections into
`data/labels/<id>.json` as suggestions, so step 8 becomes reviewing and
adjusting boxes in `ui/label.html` instead of drawing every one from
scratch. `--model` defaults to `qwen3-vl-plus` (it errors out and lists the
available enabled models if that one isn't configured); `--rung` defaults
to the highest rung each image supports, and can be overridden per run. It
never touches human work: any image whose label file is already marked
done, or already has one box a human drew or edited, is skipped; a label
file made up entirely of unreviewed suggestions gets refreshed on the next
run.

Cost: about 40 calls (one per manifest image in the bundled dataset), under
$1 at `qwen3-vl-plus`'s pricing in `configs/models.yaml`.

In `ui/label.html`, suggested boxes render dashed amber with a "?" prefix
on the tag. Edit any field on one to accept it, press `N` to accept every
remaining suggestion on a frame and move on, or `Backspace` to reject a
selected one.

One caveat: whatever model you pre-label with has its box geometry carried
into the truth set wherever you accept a suggestion instead of redrawing
it, which can flatter that same model's IoU scores relative to the others
being benchmarked. Review every box rather than mass-accepting. As a
provenance marker, `bench prelabel` writes `"labeler": "prelabel:<model>"`
into each file it touches. The first time you save that frame from the
labeling UI (an edit, a rejection, or just pressing `N`), `labeler` reverts
to your own name as usual, but the file also gains a `"prelabeled_by":
"<model>"` field that the UI keeps carrying forward on every later save,
including once the frame is marked done. Provenance survives human saves
through that field, so you can always tell which images in a finished
truth set started from a machine pass.

## Use your own brands

The bundled `brands/delay.yaml` covers five brands from the Delay Sports
corpus, but the config, labeling UI, and scoring all work against any
brands you supply:

1. Create `brands/<yours>.yaml` (copy `brands/delay.yaml` as a starting
   point): one entry per brand with `name`, a `description` used in the
   model prompt, and an optional `refs` list of reference image paths
   (relative to the repo root, typically placed under `data/refs/`) for
   brands that get confused with similar crests or wordmarks.
2. Point `configs/benchmark.yaml`'s `brands_file` at your new file.
3. Re-run `python -m bench ladder` to regenerate `data/brands.json`, which
   `ui/label.html` reads to populate the brand dropdown and the `1`-`9`
   keyboard shortcuts. If `data/brands.json` is missing or empty, the
   labeling UI refuses to load an image and tells you to run this command.

Brand names should be lowercase: model output is matched against them
casefolded, so write `name` in the yaml as lowercase to keep it consistent
with what gets reported.

## The review loop

Ground truth labels can have mistakes, and the review loop is how you catch
and correct them:

1. `python server.py` (or `python -m bench serve`) to start the local server.
2. Open `ui/review.html`. It walks through every frame where a model's
   detections disagree with the current truth labels, with truth boxes drawn
   in green and model boxes in red.
3. For each disagreement, record a verdict: model right, truth wrong, or
   both wrong.
4. `python -m bench apply-reviews` folds the verdicts back into
   `data/labels/`, correcting the labels that were wrong.
5. Re-run `python -m bench score` and `python -m bench report` to regenerate
   the leaderboard against the corrected truth.

## Metrics glossary

- **Presence F1**: per brand, frame-level precision and recall of detected
  brands against truth brands, macro-averaged across brands into a single
  headline F1 per model per rung. Detections of brands outside the configured
  brand list are ignored entirely; they cannot match truth and do not create
  extra scored brands.
- **hit@0.3 / hit@0.5**: the fraction of truth boxes matched by a detection
  of the same brand at IoU >= 0.3, and separately at IoU >= 0.5, using a
  greedy best-confidence-first match.
- **Mean IoU**: the average IoU of matched detection/truth pairs (matched at
  the >= 0.3 threshold).
- **Size and placement accuracy**: on matched pairs, the fraction where the
  model's reported size (small/medium/large) or placement
  (foreground/background) equals the truth label.
- **Retention**: a model's metric (presence F1 or hit@0.3) at a given rung,
  divided by that same model's metric at that image's highest available
  rung (1080p for full-size sources, the native height bucket for images
  smaller than 1080p). This is the resolution-robustness curve.
- **Cost per frame**: actual input and output tokens used, multiplied by the
  model's `price_in`/`price_out` from `configs/models.yaml`, averaged per
  frame.
- **Parse-fail rate**: fraction of calls whose response could not be parsed
  as the detection JSON even after the one automatic re-ask; raw result rows
  also carry a `retried` flag if you want first-attempt strictness.

The gpt-5.6 family and the Claude 5 family both reject an explicit
temperature, so OpenAI and Anthropic rows run at the API default temperature
while DashScope and OpenRouter models run at temperature 0; treat
cross-provider comparisons accordingly.

## Privacy

Images and raw model results never leave your machine except as API calls to
the providers you configure in `.env`. `data/images/`, `data/rungs/`,
`results/raw/`, and `results/gallery/` are all gitignored and are never
committed; only the manifest, labels, and aggregate `results/scores.json` /
`results/leaderboard.html` are meant to be committed.

## Building your own dataset

`scripts/build_delay_dataset.py` builds the bundled 40-frame Delay Sports
dataset from Leon's private source repo, and will not run against anyone
else's setup. Treat it as a template: it shows the pattern (select frames,
resize or extract at native resolution, write `data/manifest.json`) for
writing your own dataset-building script against your own source of images.
