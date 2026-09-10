# More query-specific capacity did not fix transfer

The source-calibrated query-aware model reaches macro nDCG@10 **0.3808**, compared with **0.3858** for the archived source router, **0.3957** for the original query, and **0.4212** for RRF across all five actions. The existing router is reconstructed exactly in every environment; in these folds its source calibration selects the fixed action prior.

| Environment | Existing router | Query-aware selection | Original | HyDE | RRF-all |
| --- | ---: | ---: | ---: | ---: | ---: |
| FiQA / BM25 | 0.2525 | 0.2525 | 0.2525 | 0.1814 | 0.2772 |
| FiQA / dense | 0.3325 | 0.3393 | 0.3921 | 0.3325 | 0.4095 |
| NFCorpus / BM25 | 0.2451 | 0.2466 | 0.2451 | 0.2116 | 0.2665 |
| NFCorpus / dense | 0.2348 | 0.1984 | 0.2348 | 0.2313 | 0.2362 |
| SciFact / BM25 | 0.6061 | 0.5834 | 0.6061 | 0.5403 | 0.6511 |
| SciFact / dense | 0.6436 | 0.6644 | 0.6436 | 0.6795 | 0.6866 |
| Equal-family/backend macro | 0.3858 | 0.3808 | 0.3957 | 0.3628 | 0.4212 |

Source calibration selects kNN in five folds and the global mean in FiQA/BM25. The five selected kNN models improve source-calibration nDCG over the fixed prior by 0.0083–0.0295. Those improvements do not reliably transport: the deployed changes against the archived router range from −0.0364 on NFCorpus/dense to +0.0208 on SciFact/dense. The positive per-environment paired intervals all include zero. These are conditional descriptive intervals on reused development data, without adjustment for repeated research use or multiple comparisons.

The selected model beats RRF-all in none of the six environments. RRF is more expensive: six searches and one shared generation per task. The query-aware selections average about 1.03–1.09 searches in their five active folds, plus generation when selecting a rewritten action and an extra query encoder. FiQA/BM25 stays fixed and avoids that encoder. This comparison does not establish quality at equal total serving cost.

There is substantial action-oracle headroom: macro per-query oracle nDCG is **0.5092**, versus **0.4047** for the target-best fixed-action oracle. This diagnostic removes the particular restriction to a shared environment posterior and four query buckets: kNN can make different predictions for every query. It does not prove that its frozen embedding representation, small source sample, or chosen model family can identify the useful action on a new corpus.

The next bottleneck is therefore **transporting query-specific action utility**, not merely allowing a router to vary its action. The source calibration split tests query generalization within familiar corpora and can reward a relationship that fails on another corpus. More source-family coverage and genuinely held-out-family calibration would directly test that failure mode. Search-result-conditioned evidence is another possible direction, with its costs and target-information contract made explicit. Additional probe selectors are premature until an observation-to-utility model beats these stronger static and inference-time baselines.

All 15 candidate outcomes remain in `all_results.json`; target outcomes selected none of the reported primary models. No conclusions about arbitrary query-aware models, new target families, or a new algorithm follow from this limited exploratory grid.
