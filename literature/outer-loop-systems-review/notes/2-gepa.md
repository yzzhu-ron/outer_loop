# GEPA — Reflective Prompt Evolution

- **Paper:** "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (arXiv:2507.19457), Agrawal et al.; ICLR 2026. OpenReview: https://openreview.net/forum?id=RQm2KQTM5r
- **Repo:** https://github.com/gepa-ai/gepa (cloned in `../2-gepa/gepa/`; MIT; also `dspy.GEPA`)

## Core idea

GEPA (Genetic-Pareto) optimizes any *textual* parameter of a system — prompts, code, configs — via an evolutionary loop where mutation is done by an LLM that *reads full execution traces* (errors, logs, reasoning) instead of collapsing them to a scalar reward. Selection is Pareto-aware: it keeps candidates that win on *any* training instance, not just the best average, preserving complementary lessons and avoiding local optima.

Loop: sample system trajectory → reflect in natural language on what failed → propose targeted prompt/text mutation → evaluate → keep on Pareto frontier → optionally merge complementary candidates.

## Key results

- Beats GRPO (RL) by ~10% avg, up to 20%, with up to **35x fewer rollouts** (100–500 evals vs 5k–25k+).
- Beats MIPROv2 (prior best prompt optimizer) by >10%.
- `optimize_anything` API extends beyond prompts: ARC-AGI agent 32%→89% via architecture discovery; cloud scheduling policy beating expert heuristics (40.2% cost savings); coding-agent skills 55%→82% resolve rate on Jinja tasks.

## Impact / community feedback

- Heavy production adoption: 50+ uses across Shopify, Databricks (90x cheaper than Claude Opus 4.1 for enterprise agents), Dropbox, OpenAI, Pydantic, MLflow, Comet ML.
- Tobi Lütke (Shopify CEO): "DSPy and (especially) GEPA are currently severely under hyped."
- Ecosystem: GEPA-Lite, gepa-mcp (Claude Desktop integration), Arize benchmark comparisons, Decagon production guide.
- Criticisms / limitations seen in the wild:
  - Overfitting to the optimization set: test performance can peak after a few steps then degrade (seen in deception-detection and Verilog-generation studies); needs careful val/test splits.
  - Works best when traces carry rich textual feedback; weak-signal domains gain less.
  - DSPy-format requirement adds friction for existing systems (mitigated by `optimize_anything`).

## Relevance to ShopGym

- The most "drop-in" of the four works: ShopGym's LLM-judge + binary verifier already produce the metric; the harness's system prompt, observation formatting instructions, and action-space description are textual parameters GEPA can evolve directly.
- ShopGuru's per-task feedback (verifier findings, judge rationales) is exactly the "actionable side information" (`oa.log`) GEPA reflection consumes.
- Pareto selection over *task categories* (search-exact, filter, long-horizon journeys…) would preserve harness variants that excel at different skill groups — matching ShopGuru's skill-catalog structure.
- Caution: with ~224 tasks, hold out a test shop (or generate fresh shops/tasks — ShopGym's unique advantage) to detect prompt overfitting.
