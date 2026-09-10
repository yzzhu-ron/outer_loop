# Full-paper figure notes

These figures describe controlled mechanisms and development experiments; they do not establish untouched-family transfer.

Counts: 1,658 controlled result rows; 14 plotted canonical rows. Natural retrieval has 1,620 run rows, 60 baseline rows, and 18 primary lookahead runs. Fusion has 144 environment configurations, displayed as 24 equal-environment macro cells.

SciFact and FiQA each contribute 150 query identities; NFCorpus contributes 65. The same queries run against two backends: 365 family-specific identities and 730 query–backend evaluations per method. Seeds and backend configurations are correlated; no macro confidence intervals are plotted.

## Controlled mechanisms

Files: controlled_mechanism_v1.svg, controlled_mechanism_v1.png.

Two horizontal bar charts compare seven methods at probe budget two and workload weight three quarters. In both constructed mechanisms, no probes score 0.5, information gain 0.5625, decision-region entropy 0.625, and myopic value 0.75. Two-step lookahead and full-horizon planning score 0.875 in both. The best source-fixed plan scores 0.875 for XOR and 0.75 for the gated mechanism. These are exact expected utilities, not empirical confidence intervals.

## Natural retrieval and capacity

Files: natural_capacity_v1.svg, natural_capacity_v1.png.

A six-environment dot plot compares the source prior, original query, primary two-step lookahead at budget 64, all-action reciprocal-rank fusion, and a target-label router-capacity bracket. Lookahead minus prior: SciFact · BM25 -0.0231; SciFact · dense +0.0195; FiQA · BM25 +0.0034; FiQA · dense -0.0094; NFCorpus · BM25 -0.0018; NFCorpus · dense +0.0003. RRF exceeds lookahead in all six environments and exceeds the capacity upper bound in five; NFCorpus dense is the exception. The SciFact BM25 capacity bracket is 0.61846 to 0.62445; the other five brackets collapse to a verified value within numerical tolerance. The plots reuse 365 queries across two backends and report descriptive seed means, with no macro confidence interval.

## Fusion quality and serving searches

Files: fusion_frontier_v1.svg, fusion_frontier_v1.png.

Two line charts show equal-family/backend mean retrieval quality against actual mean search calls per task, with backend identity known or hidden. With generation allowed and the six-call cap, known-backend source selection achieves 0.41402 nDCG with 4.333 mean searches; hidden-backend selection achieves 0.41285 with 4.667. All-action RRF achieves 0.42118 with six searches. Source selection without generation uses at most 1.333 mean searches and stays near 0.39 nDCG. Curves connect the recorded source-selected configurations, with no target Pareto filtering or macro confidence intervals. Generation and token costs are separate from the plotted search axis.

## Query-only transfer diagnostic

The separate known-backend query-transfer run contains 60 primary rows, 150 total result rows, and 90 source calibration rows. It uses the same 365 queries/730 query–backend evaluations. Candidate selection holds out source queries within known source corpora; it is not held-out-family validation.

Equal-family/backend macro nDCG@10: source-selected query router 0.38076; source-selected kNN 0.38078; reconstructed existing router 0.38577; original 0.39571; all-action RRF 0.42118. The reconstructed router reproduces every archived action. These known-backend scores are a separate diagnostic from the hidden-backend natural acquisition comparison.

## Provenance

Exact input hashes, plotted values, selection rules, and generation/token costs are in `publication_figure_manifest.v1.json`. The capacity audit is a post-hoc target-label diagnostic: floating-LP upper bounds and verified deterministic-policy lower witnesses, not statistical intervals.
