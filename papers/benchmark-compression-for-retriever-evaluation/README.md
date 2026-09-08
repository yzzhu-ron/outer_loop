# Benchmark Compression for Retriever Evaluation

Status: prospective paper direction. The benchmark and baseline release are
complete; the paper-specific outer-loop study described here has not yet been
run.

## Paper in one sentence

Learn a small, nested set of evaluation queries that preserves the conclusions
of a 40,159-query retriever benchmark—including model ordering, nDCG estimates,
and winners—when evaluated on retriever families absent from development.

The title names the scientific object, the intervention, and the evaluation
domain directly.

## Research question

Full retriever evaluations are expensive enough to constrain how many systems
an autonomous research loop can compare. Randomly reducing the queries saves
work but may change the conclusion, especially when systems are close. The
paper should ask:

> Can structured-feedback-guided artifact search construct compact, nested query sets that
> preserve retriever comparisons and transfer to unseen retriever families
> better than random sampling, clustering, and direct visible-score matching?

The artifact is a deterministic selector, or equivalently its ordered query
priority. The evaluator supplies structured residuals: which datasets,
retriever pairs, score estimates, and winners are currently misrepresented.
The proposer adds, removes, or swaps queries in response. Selection operates on
held-out visible families and retains candidates that improve fidelity without
violating exact size, nesting, coverage, or uniform-weight constraints.

## Existing benchmark record

The public task release is maintained independently in
[`yzhu319/agentic_research_benchmarks`](https://github.com/yzhu319/agentic_research_benchmarks),
under `benchmarks/benchmark-compression-optimization/`. It already provides:

- 40,159 source-backed queries across 13 BEIR datasets;
- visible per-query nDCG@10 outcomes from six retriever families;
- exact nested budgets of 650, 1,300, and 3,250 queries;
- an evaluator combining pair-order, score-estimation, and winner fidelity;
- random, clustering, Anchor Points, tinyBenchmarks, and visible-mean-matching
  baselines; and
- one frozen, one-shot measurement on three sealed retriever families.

The frozen visible-mean-matching baseline scores `0.979940` on the visible
panel and `0.8746722583361243` on the sealed families. That gap is the central
scientific opening: fitting the observed retrievers is easier than choosing
queries that transfer.

The benchmark repository owns the data, evaluator, selector, release package,
and technical documentation. This directory owns the paper question,
experimental design, and eventual manuscript. A paper experiment can consume a
pinned public release just as it would consume BEIR or another external
dataset.

## What would make this an outer-loop paper

The released benchmark alone is a benchmark contribution. The paper becomes an
outer-loop research contribution only if it studies the process used to improve
the selector and shows that structured feedback changes search efficiency or
generalization.

The minimum useful comparison is:

| Method | Proposal signal | Selection signal | Purpose |
|---|---|---|---|
| Equal-allocation and stratified random | none | none | data-independent controls |
| k-medoids / Anchor Points / tinyBenchmarks | feature or outcome geometry | one fitted solution | prior benchmark-compression methods |
| Visible-mean matching | visible aggregate residual | greedy deterministic updates | strong task baseline |
| Unconditioned outer loop | random valid edits | held-out-visible fidelity | separates search from feedback value |
| Feedback-guided greedy loop | dataset/pair residuals | single incumbent | measures feedback value and local traps |
| Feedback-guided archive loop | dataset/pair residuals | Pareto archive across datasets and budgets | tests diversity under conflicting objectives |

The key ablation holds the artifact space and evaluation budget constant while
varying structured error feedback and greedy versus archive selection.

## Proposed evaluation protocol

1. Freeze the public benchmark release and never use the sealed-family outcome
   rows for development. The existing one-shot hidden score remains a baseline
   fact, not a tuning channel.
2. Partition the six visible families into nested development and validation
   roles. Rotate the held-out visible family so every proposal method is judged
   out of fit at least once.
3. Give every method the same number of candidate evaluations and the same
   exact query budgets. Record selector runtime separately from retrieval cost,
   which is already represented by the frozen outcome matrix.
4. Report fidelity at each budget, each dataset, and the equal-dataset macro
   level. Also report worst-dataset fidelity and variance across family-holdout
   folds; a high mean must not hide a collapsed dataset.
5. Pre-register one final sealed evaluation of the chosen method only if a new
   untouched hidden panel can be provisioned. Do not reuse the existing sealed
   families as an iterative paper-development test set.

## Candidate paper claims

These are the hypotheses to register before running the paper study:

1. Structured dataset/pair residuals increase the probability that a proposed
   selector improves held-out-family fidelity over an unconditioned edit.
2. A Pareto archive improves worst-dataset and small-budget fidelity over a
   greedy single-incumbent loop under the same evaluation budget.
3. Validation across retriever families reduces the visible-to-sealed transfer
   gap compared with direct visible-mean matching.
4. The learned 650-query tier preserves the qualitative conclusions of the
   full evaluation while removing 98.38% of the queries in this task setting.

Claim 4 is scoped to the released Liquid NanoBEIR-seeded,
relevance-complete reduced-corpus design. Official NanoBEIR and full-corpus
BEIR require separate validation.

## Why this is a credible ICLR 2027 direction

The artifact and evaluator are compact, deterministic, and already packaged.
The central failure—excellent visible fit but weaker transfer to unseen
retriever families—is concrete and scientifically legible. This makes it
possible to study proposal feedback, archive selection, validation leakage,
and generalization with controlled evaluation cost.

The tradeoff is equally clear: benchmark compression is a narrower artifact
domain. A strong submission needs a real algorithmic or empirical result beyond
releasing the task. If the outer-loop variants do not beat the frozen baseline
under family-held-out validation, the honest outcome is that the benchmark is a
useful negative test for the framework, not a paper claim to stretch.

## Immediate next work

1. Define a valid local edit language over nested selections.
2. Convert evaluator failures into proposer-readable residual summaries without
   exposing the held-out family used for selection.
3. Implement unconditioned, greedy, and archive loops against the pinned public
   release.
4. Run family-held-out experiments with a fixed candidate-evaluation budget.
5. Decide on the ICLR direction from transfer performance, not from title
   preference alone.

## Primary sources

- Thakur et al., [BEIR](https://arxiv.org/abs/2104.08663), NeurIPS 2021.
- Kamalloo et al., [Resources for Brewing BEIR](https://arxiv.org/abs/2306.07471), SIGIR 2024.
- Vivek et al., [Anchor Points](https://aclanthology.org/2024.eacl-long.95/), EACL 2024.
- Maia Polo et al., [tinyBenchmarks](https://proceedings.mlr.press/v235/maia-polo24a.html), ICML 2024.
- Carterette et al., [A Few Good Topics](https://doi.org/10.1145/1629096.1629099), SIGIR 2009.
- The pinned [Liquid NanoBEIR multilingual extended dataset](https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended/tree/8a4be55eb80b3ed4d2e9a423a5212228c217d426).
