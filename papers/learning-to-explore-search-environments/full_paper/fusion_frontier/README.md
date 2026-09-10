# Source-selected fusion frontier

This is a deliberately strong simple baseline, not a new method. It evaluates all 31 nonempty subsets of the five original query actions, then chooses a fixed subset using only source training queries. Choices are constrained by 1–6 retrieval calls and whether generation is allowed. The [protocol](protocol.json), code, and minimized evidence were frozen in commit `b2ed2fa` before computing this analysis.

For hidden-backend source fitting, the two-search point reaches .409352 macro nDCG@10, compared with .395706 for original-query retrieval and .421179 for all-action RRF at six searches. Both fusion points use one shared generation call per task. This is a quality/cost tradeoff, not a demonstration of equal quality. Results vary by family; the two-search source choice hurts FiQA/dense relative to its original query. No onboarding is used.

[results.json](results.json) preserves 144 source-selected configurations, every target subset score explicitly labeled evaluator-only, paired query outcomes, selected subsets, and token/call costs. Same-backend source fitting uses two worlds; hidden-backend fitting uses four. Three correlated corpus families remain the units for generalization.

For a singleton, keep its existing ranked list. For larger subsets, apply RRF(k=60) to the available top 10 action lists, then truncate to 10. This is **truncated-list fusion**, not a rerun at full retrieval depth. The decomposed action already fuses its two subqueries. Generation produces semantic, HyDE, and decomposed variants in one response: searches sum across selected actions, while shared generation/token usage is counted once.

To reanalyze on another machine, copy this directory and the adjacent `natural/experiment.py` (a hash-checked metrics dependency). With NumPy installed, move the existing results file aside in your copy, then run:

```sh
python experiment.py
python -m unittest -v test_frontier
```

The executable refuses to overwrite `results.json`. Its `--export` mode requires the original semantic caches, but is unnecessary for replay: `evidence.json` already contains minimized source/test rankings, qrels, and costs. Replay computes fusion scores from those recorded rankings; it does not rerun the underlying retrieval systems. Timing depends on the machine; all method choices are deterministic under the recorded tie rule.
