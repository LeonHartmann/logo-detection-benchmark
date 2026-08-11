# Tools that could make vision models score better

Research note, 2026-08-11. What "tools" (agentic capabilities beyond a single
look at the image) are worth adding to this benchmark, ranked by expected
value for logo detection in social media screenshots. Grounded in our own
24-model results and current literature.

## Ranked recommendations

### 1. Zoom / crop-and-look-again (IMPLEMENTED)

The model may request an enlarged crop of the target before answering
(`tools: [zoom]` on a model row, up to 3 zooms per frame). This is the
single best-supported tool in current research: CropVLM trains zooming with
reinforcement learning and shows fine-grained perception gains without any
box supervision (arXiv 2511.19820); follow-up work disentangles why
crop-and-zoom helps (arXiv 2602.01334); zooming also unlocks large gains in
GUI grounding (arXiv 2512.05941); and multi-resolution zoom encoding
(Dragonfly, arXiv 2406.00977) shows the same effect baked into the
architecture. OpenAI's reasoning models already zoom internally ("thinking
with images"), which may explain their strong low-rung retention in our
results.

Design rule we enforce: zoom serves crops of the SAME resolution rung,
upscaled 2x. It never reveals pixels the rung does not contain, so the
resolution-robustness axis stays honest. What it tests: whether low-rung
failures come from attention and tiling limits (zoom fixes those) or from
genuinely missing pixels (zoom cannot).

Expected gain: biggest at 240p and 144p, and for frontier generalists whose
presence F1 is high but whose box quality is poor. Cost: 2-4x tokens and
latency per frame.

### 2. Specialist-detector tool (the cascade, formalized)

Give a reasoning model a `detect(brand_description)` tool backed by a
grounding specialist (qwen3-vl family here; open-vocabulary detectors like
Grounding DINO or OWL-ViT/Florence-2 in the local-model world). The
generalist decides what to look for and filters false positives; the
specialist draws the boxes. Our own leaderboard is the argument: the best
generalist (claude-opus-5, presence F1 0.799) hits only 0.143 of boxes at
IoU 0.3, while qwen3-vl-plus hits 0.603 with mean IoU 0.886 at 1/40th the
price. "Agentic object detection" (LandingAI, 2026) productizes exactly
this pattern. This is also the ensemble cascade: implement once, report as
a condition row.

Expected gain: combines the best presence scores with the best boxes;
likely the strongest overall condition. Cost: one extra cheap call per
frame plus orchestration.

### 3. OCR tool for wordmark brands

Three of our five brands are text (dkh, 11teamsports, the delay wordmark
variant). A `read_text()` tool backed by a proper OCR engine (PaddleOCR
class) returns word boxes the model can reason over; text spotting at 144p
is exactly where VLMs degrade fastest. Expected gain: concentrated on
11teamsports (our worst brand: best model reaches F1 0.40) and dkh at low
rungs. Cost: local, near-free per frame.

### 4. Tiling / set-of-marks sweep (a zoom variant for single-turn models)

Pre-cut the frame into overlapping tiles, let the model answer per tile,
merge like the ensemble union. Works for models that cannot hold a
multi-turn conversation (our two format-refuseniks) and needs no tool
protocol at all. Research lineage: set-of-marks prompting and ViCrop-style
attention crops. Expected gain: similar direction as zoom but bounded;
costs a fixed 4-6x calls per frame, no model cooperation needed.

### 5. On-demand reference lookup

A `get_brand_reference(brand)` tool that returns the reference crop only
when the model asks. Saves the token overhead of always-attached refs and
tells us whether models know when they are unsure. Cheap to build once refs
exist; expected gain small but the metadata (who asks, when) is
diagnostically interesting.

### Not recommended

- Super-resolution/enhancement tools (Real-ESRGAN class): they hallucinate
  texture at exactly the scale where logos live; a benchmark about truth
  should not upsample evidence beyond the honest 2x Lanczos zoom.
- Color-histogram or metadata probes: nothing in our failure gallery
  suggests color confusion is a failure mode.

## Sources

- [CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](https://arxiv.org/abs/2511.19820)
- [What Does Vision Tool-Use RL Really Learn? Crop-and-Zoom effects](https://arxiv.org/pdf/2602.01334)
- [Zoom in, Click out: Zooming for GUI Grounding](https://arxiv.org/pdf/2512.05941)
- [Dragonfly: Multi-Resolution Zoom-In Encoding](https://arxiv.org/pdf/2406.00977)
- [Best Object Detection Models 2026 (agentic object detection overview)](https://www.ultralytics.com/blog/best-object-detection-models)
- [Top Vision Language Models 2026](https://www.datacamp.com/blog/top-vision-language-models)
