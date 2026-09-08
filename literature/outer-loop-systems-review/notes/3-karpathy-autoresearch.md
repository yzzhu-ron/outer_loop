# autoresearch — Karpathy's autonomous research loop

- **Author:** Andrej Karpathy, released March 7, 2026.
- **Repo:** https://github.com/karpathy/autoresearch (cloned in `../3-karpathy-auto-research/autoresearch/`)

## Core idea

Point a coding agent at a minimal single-GPU nanochat training setup and let it experiment overnight: read `train.py`, propose a change, train for a fixed 5-minute wall-clock budget, measure `val_bpb`, keep if improved / revert if not, log to `results.tsv`, repeat. ~12 experiments/hour, ~100 while you sleep.

Deliberately minimal — three files matter:

- `prepare.py` — frozen: data, tokenizer, dataloader, eval. **The harness/metric is immutable.**
- `train.py` — the only file the *agent* edits.
- `program.md` — the only file the *human* edits ("you are programming the research org, not the Python").

Design choices worth noting: fixed time budget makes all experiments comparable regardless of what changes; single metric (val_bpb, vocab-independent); explicit *simplicity criterion* in program.md ("an improvement of ~0 but much simpler code? Keep") — a built-in compression pressure, same instinct as HL's "compress history."

## Impact / community feedback

- Explosive: 66k+ GitHub stars and 9.6k forks within a month. Fortune dubbed it "The Karpathy Loop" (700 experiments in 2 days).
- Community forks: macOS MLX (val_bpb 2.667 → 1.294 overnight on M4 Max), Windows/RTX, TinyStories small-compute variants; an awesome-autoresearch list applies the ratchet to prompt optimization, coding-skill improvement, and software perf benchmarks.
- "AutoAgent"-style derivatives apply the same loop to *agent harnesses*: edit the harness, hill-climb on benchmark scores — exactly the ShopGym use case.
- Criticisms:
  - "Rediscovered AutoML" — labs have done automated architecture/hyperparameter search for years; the novelty is the LLM proposer + dirt-simple ratchet, not the concept.
  - Goodhart's Law made executable: agents exploit the metric (e.g., a Gomoku fork where the agent replaced the NN with handwritten alpha-beta search, 99.3% win rate — which is also, amusingly, a point *for* HL).
  - Unverified community claims spread fast (a "53% speedup" Shopify-attributed claim stayed unmerged, flagged as overfit to eval noise) — noisy 5-min runs invite false positives; needs repeat-seed verification.
  - Platform-specific results don't transfer; no error recovery beyond what program.md specifies.

## Relevance to ShopGym

- The cleanest *protocol* template: one mutable surface (harness code), one frozen evaluator (ShopGuru verifier suite), one metric (success rate), fixed eval budget per iteration, append-only results log, keep/revert ratchet.
- The `program.md` pattern: encode your harness-research instructions, constraints (e.g., "do not modify the judge or the tasks"), and simplicity criterion in one human-edited markdown file the optimizing agent reads.
- The Goodhart lesson is critical for ShopGym: the LLM-judge is gameable; keep rule-based verifiers as ground truth where possible and freeze them like `prepare.py`. Repeat runs (ShopGym already aggregates over repeats) guard against eval-noise false positives.
