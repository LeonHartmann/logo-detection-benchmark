# Logo Detection Benchmark: Management Summary

Research by Leon Hartmann · 2026-08-12

## Purpose

Sponsorship reporting depends on detecting partner logos in social media
content at scale. This benchmark answers, with measured evidence: which
vision AI models detect sponsor logos reliably, how accuracy degrades with
image resolution, what each option costs, and which prompting methods help
or hurt.

## Method

- 39 real frames from the Delay Sports corpus, all sponsor logos
  human-labeled (242 boxes, 5 brands: adidas, stripes, dkh, 11teamsports,
  delay), each frame tested at 5 resolutions (1080p to 144p).
- 24 models from 10+ vendors, one identical prompt, plus 18 method variants
  (reference images, an interactive zoom tool, per-brand calls).
- 8,400 scored results, 42 leaderboard rows, $143 total API cost. Full
  harness, data definitions and interactive dashboard are reproducible and
  published (github.com/LeonHartmann/logo-detection-benchmark).

## Headline results

| Question | Answer |
|---|---|
| Best at finding brands | claude-opus-5 and gemini-3.6-flash+zoom (F1 0.799); muse-glimmer-30b with per-brand calls leads overall at 0.835 |
| Best at localizing (boxes) | qwen3-vl-plus by a wide margin: 60% of boxes hit at $0.80 per 1,000 frames; frontier models land only 1-14% |
| Best value | seed-2.0-mini: near-frontier detection at $1 per 1,000 frames (frontier equivalent: $36) |
| Resolution robustness | Gemini retains 93% of accuracy at 144p; the box specialist retains 58% |
| Methods | Zoom tool: modest real gains. Reference images: halve box accuracy. Per-brand calls: hurt 5 models, helped 1. Method effects are model-specific |

## Business implications

1. **Cost**: production-grade logo detection costs roughly $2 per 1,000
   frames using a two-model cascade (cheap detector for presence, specialist
   for boxes), not the $36+ per 1,000 that frontier models charge. At
   full-corpus scale (20k+ frames per scan) this is the difference between
   negligible and material cost.
2. **Quality**: the recommended cascade matches or beats the current
   production engine (qwen3.8-max) on detection while improving box quality
   and cutting cost by more than 90 percent.
3. **Risk**: model rankings measured at full HD do not transfer to low-res
   content. Pipelines ingesting thumbnails must benchmark at thumbnail
   resolution.
4. **Vendor independence**: the harness is model-agnostic; new models are a
   one-line config entry, so pricing or capability shifts can be re-evaluated
   within an hour for a few dollars.

## Recommendation

Adopt the cascade (seed-2.0-mini with zoom for presence, qwen3-vl-plus for
boxes) as the candidate next-generation scan engine, validated against the
existing production pipeline on a full corpus scan before switching.
Reference images should not be added to any box-drawing stage.

## Caveats

- One brand (11teamsports) has a single ground-truth frame; its per-brand
  figures are indicative only. Extending truth data for this brand is the
  main open data task.
- OpenAI gpt-5.6 and Anthropic Claude 5 APIs reject fixed randomness
  settings and ran at provider defaults.
- Ground truth was machine-pre-labeled and human-verified box by box;
  provenance is recorded per label.

## Next steps

1. Ensemble evaluation (voting and cascade variants) from already-collected
   data, zero additional API cost.
2. Specialist-detector tool condition (reasoning model calls a grounding
   model), the strongest remaining improvement candidate from the research.
3. Additional 11teamsports truth frames; disagreement review loop to refine
   labels.
