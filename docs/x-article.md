# X Article: paste-ready draft

Cover image: `results/article/ranked.png` (or `hero.png` for a stats-tile cover).
Image markers show where to insert each PNG from `results/article/`.
All numbers are from the final 42-row matrix, 2026-08-12.

---

## Title

**Which AI can actually spot a sponsor logo? I tested 24 models on 8,400 detection tasks to find out**

## Body

Sponsorship money follows visibility. If your club's crest shows up in ten
thousand fan photos and match clips, someone has to count those logos, and
increasingly that someone is an AI. So I built a benchmark to answer a simple
question: which vision models can actually do this job, and at what price?

The setup: 39 real social media frames from a football club's channels,
every sponsor logo hand-labeled by me (242 boxes across 5 brands), each image
served at 5 resolutions from crisp 1080p down to 144p thumbnail mush. Every
model gets the same prompt and has to name the brand AND draw a box around
it. 24 models from OpenAI, Anthropic, Google, Meta, Alibaba, ByteDance,
Mistral, xAI and more, plus 18 method variants. Total damage: $143 in API
calls.

Here is what surprised me.

**1. Nobody wins everything. Not even close.**

[IMAGE: results/article/ranked.png · Top models by presence F1 at 1080p]

The best models at KNOWING a logo is present are frontier heavyweights and
one shocking newcomer. But ask the same models WHERE the logo is and the
ranking flips completely.

Claude Opus 5 finds brands at 0.80 F1, best single-call score in the field.
Its boxes? Only 14 percent land on the actual logo. Google's Gemini: same
story. Anthropic's Sonnet: under 1 percent. **Frontier models know a logo is
there but cannot point at it.**

Meanwhile a specialist, Qwen3-VL-plus, hits 60 percent of boxes with 0.89
mean IoU, at 80 cents per 1,000 frames. That is not a typo. The best
box-drawer in the benchmark costs less than a coffee per ten thousand images.

**2. Price predicts almost nothing.**

[IMAGE: results/article/pareto.png · Cost vs quality, log scale, Pareto frontier in orange]

ByteDance's seed-2.0-mini scores 0.76 at finding brands. Claude Opus 5
scores 0.80. The price difference: $1 versus $36 per 1,000 frames. A model
costing 36x more buys you four hundredths of a point. And the most expensive
model per answer in my whole test (kimi-k3 at $93 per 1,000 frames, it
thinks for 95 seconds per image) lands mid-table.

**If you are paying frontier prices for detection workloads, you are
probably overpaying by an order of magnitude.**

**3. Resolution is a silent killer, but not for everyone.**

[IMAGE: results/article/retention.png · presence F1 from 1080p down to 144p]

At 144p, Gemini keeps 93 percent of its 1080p performance. The box
specialist Qwen3-VL-plus keeps 58 percent. **Specialists need pixels;
Gemini reads tea leaves.** If your pipeline ingests low-res thumbnails, the
model ranking you benchmarked at full HD is lying to you.

**4. The "obvious" improvements mostly backfired.**

This is where it gets fun. I tested three method upgrades everyone would
bet on.

[IMAGE: results/article/method-presence.png · base vs refs vs zoom vs per-brand, finding brands]

**Reference images (showing the model what each logo looks like): mostly a
trap.** Presence detection barely moved, but box accuracy HALVED for the
box-strong models. Five reference collages in the prompt drowned the target
image and confused coordinate systems.

**A zoom tool (the model may request enlarged crops before answering):
modest, real gains.** It pushed Gemini Flash into a tie for first place at a
third of Opus's price, and it doubled Opus's box accuracy. Models also used
it rationally: twice as many zooms at 144p as at 1080p.

[IMAGE: results/article/method-boxes.png · the same comparison for box accuracy]

**Per-brand calls (one focused question per brand): wrecked five models and
crowned one.** Four of five models fell hard, one small Meta model
(muse-glimmer-30b) jumped from 0.79 to 0.84 and took the overall lead of the
entire benchmark. Same method, opposite outcomes. **Method effects are
model-specific. Test before you standardize.**

**5. Raw intelligence is worthless if the model cannot follow a format.**

Two models were effectively unusable, not because they cannot see, but
because they answer in their own dialect instead of the requested JSON. One
failed to parse on 75 percent of calls. In production, format discipline IS
capability.

**What I would actually deploy**

A cascade of two cheap models: seed-2.0-mini with the zoom tool to find
brands (0.79 F1 at $1.20 per 1,000 frames) and Qwen3-VL-plus to draw the
boxes (60 percent hits at $0.80). Roughly $2 per 1,000 frames, near
frontier-level quality. The $36 frontier model is the wrong tool for this
job.

**The honest fine print**

One of my five brands has a single ground-truth frame, so its per-brand
numbers are anecdotes. OpenAI's and Anthropic's newest models reject a fixed
temperature, so they ran at API defaults. Truth was pre-labeled by one model
and human-reviewed box by box, which may slightly flatter it. Every number,
every disagreement image, and the full harness are open: you can rerun all
of it on your own brands with your own images.

**The takeaway**

Stop asking "which AI is best". Ask "best at what, at which resolution, at
what price, under which prompt". In my 8,400 tests, those four questions had
four different winners.

What should I test next: an OCR tool for text logos, or detection ensembles
that vote? Tell me in the replies.

Research by Leon Hartmann

---

## Teaser post (publish a few hours before the Article)

I spent $143 making 24 AI models hunt sponsor logos in 8,400 images.

The winner beat a rival costing 36x less by 0.04 points.

Full breakdown with every chart drops later today.

## Thread excerpts (post after publishing, link back to the Article)

1. Frontier AIs know a logo is there but cannot point at it. Claude Opus 5:
   0.80 at finding brands, 14 percent at boxing them. A $0.80-per-1k
   specialist hits 60 percent.
2. Showing models reference images of the logos made box accuracy WORSE.
   It halved. Attention is a budget; five reference collages spent it.
3. A zoom tool helped exactly the way you would hope: models zoomed 2x more
   often on low-res images, and it doubled Opus's box accuracy.
4. The same per-brand method wrecked five models and made a sixth the
   overall champion. Test before you standardize.
