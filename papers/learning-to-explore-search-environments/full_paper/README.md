# Full-paper research

Start with the [technical blog draft](searchprobe-blog.html). It explains the idea, the controller failure, complementary evidence, and the new retrieval results. It is self-contained and works offline. This directory continues milestone `e330622` on `codex/searchprobe-full-paper`; the earlier protocols and evidence are preserved.

The full paper is **not finished**. The new experiments sharpen the problem, but do not establish a novel acquisition method that improves unseen search environments. The contribution we want to earn is a source-trained relationship between cheap, automatically checkable search experiments and the marginal value of later search decisions. Generic value of information, decision-region discovery, workload robustness, and lookahead are established ideas; [the literature audit](theory/LITERATURE.md) records close predecessors.

## Evidence from this cycle

| Study | What it establishes | What it does not establish |
| --- | --- | --- |
| [Controlled suite](controlled/RESULTS.md), 1,658 exact rows in 38 conditions | Immediate-value rewards can miss complementary evidence; adaptive probing can beat the best fixed set in a gated example; hidden workload shifts and misspecified observation models can reverse gains. | Retrieval effectiveness, novel VOI theory, or a universal ordering of methods. Tie sensitivities are included. |
| [Finite-channel retrieval](natural/README.md), 1,620 run rows | Two-step acquisition .3824 macro nDCG vs random .3782 and IG .3805, but below its own prior .3842 and original .3957. | A successful development gate or generalization to a fresh family. Backend identity is hidden, unlike the earlier milestone. |
| [Capacity audit](natural/capacity_audit.json) | Even the best four-bucket/five-action policy is below all-action RRF in five of six environments. A verified posterior witness and a weak-tie upper relaxation separate attainable from optimistic scores. | A bound on richer query-dependent policies or fusion. Floating LP statuses are checked; these are not exact rational infeasibility proofs. |
| [Fusion frontier](fusion_frontier/README.md) | Source-selected two-search fusion .4094 vs all-action fusion .4212 at six searches, both with one generation call and no onboarding. | Equal quality, a new fusion algorithm, or a probe-selection advantage. All operating points are retained. |
| [Query-level transfer](query_transfer/README.md) | Tests source-selected nearest-neighbor routing against the reconstructed original ridge/fixed pipeline, with no onboarding. | An unseen-family confirmation: these three corpora were already development data. |
| [Intervention-response transfer](intervention_transfer/README.md) | The within-corpus response model scores .4290 versus .4295 for same-data fixed fusion. Even the target-label global-policy oracle can gain only .00231 over that fixed baseline, below the registered .005 gate. | Evidence that probes are uninformative, a successful utility bridge, or an upper bound on query-dependent policies. These are seven engineered mixtures of the same cached endpoints. |
| [Supporting theory](theory/THEORY.md) | Exact small workload games, a sharp restricted coverage/information tradeoff, and a counterexample to changing the order of worst case and expectation. | Novelty for standard minimax, DRD, perturbation bounds, or reward-free exploration. |

The endpoint studies use the same 365 family-specific query IDs with two correlated backends: 730 query/backend cases. The intervention study evaluates those same questions at seven engineered mixture settings, giving 2,555 paired query/mixture cases, repeated across three probe panels. There are **three development families** throughout. Source utility fits and target evaluation stay separate in code, but all three families have informed research decisions. No untouched-family confirmation claim is made.

## Run the useful part in five minutes

The [SearchProbe package](../../../packages/searchprobe/README.md) contains editable synthetic examples in its installed wheel. Run these commands from the repository root:

```sh
python -m pip install ./packages/searchprobe
searchprobe-demo --output-dir editable-examples
```

The [portable evidence replay](replay/README.md) recalculates 810 saved milestone run summaries, 600 paired-contrast means, and 12 conditional model certificates from 34 checksum-pinned files. It needs only Python and SearchProbe; no GPU, MLX, NumPy, model downloads, or ignored caches. It is analysis replay, not retrieval/training regeneration.

The new natural/fusion/query-transfer experiments also commit their minimized input evidence. Their READMEs distinguish portable reanalysis from the larger process that originally generated the retrieval data. The controlled and workload examples use only the standard library; the channel diagnostics use NumPy and SciPy.

## What determines the next paper

The failures now distinguish controller responsiveness, policy capacity, and utility transfer. In the hybrid experiment, the strongest source-fixed fusion happens to match the target-best fixed policy in all three families. A different global policy for each backend offers too little additional quality to meet the registered gate, even with target labels. That finite-benchmark limit does not apply to richer query-dependent policies. More lookahead by itself is not a credible contribution.

The [posthoc audit](theory/POSTHOC_INTERVENTION_REVIEW.md) separates query routing from backend adaptation. The per-question oracle scores .5189 while keeping each question's strategy fixed across backends, versus .5286 when it can adapt. Much of the apparent opportunity therefore exists without onboarding. Four simple query categories offer .00639 of additional backend-adaptation room, but source categories are sparse and exploratory calibration gains concentrate in one family. These target-label ceilings are neither learnable policies nor future-population guarantees. A new probe-dependent method must beat an equally calibrated query-aware baseline with no probes.

The [intervention study](intervention_transfer/README.md) tested whether probe-response changes predict action or fusion utility while holding documents and relevance judgments fixed. The code was committed before generating new outcomes; target predictions were committed before generating test utilities. It retained a source-only intercept, matched source-label budgets, pooled and shuffled controls, and grouped query uncertainty. Its practical and attribution checks failed. We will not lower the gate after seeing results.

A full-paper claim requires a model class with meaningful adaptation opportunity, a transferable utility model, strong source-selected fusion and query-aware baselines, complete costs, and untouched collection families. The next design must check source-side opportunity and query coverage before spending effort on acquisition. A positive engineered example or a gain over a weak starting router will not meet that bar. Keep the current three families for development; evaluate fresh families only after the mechanism earns that expense and interpretive cost.

[PAPER_PATH.md](theory/PAPER_PATH.md) makes that next study concrete: proposed source expansion and screening gates, a fixed-panel transfer test, and acquisition comparisons only if the panel helps. It also specifies when to redirect the claim. This is a proposed design, not a registration or evidence that the proposed mechanism works.

See [VALIDATION.md](VALIDATION.md) for the tested installation, evidence replay, UI checks, and completed CI run.
