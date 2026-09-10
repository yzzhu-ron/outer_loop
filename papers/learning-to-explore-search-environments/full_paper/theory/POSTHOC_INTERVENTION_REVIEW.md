# What the registered intervention result rules out

**POSTHOC evaluator audit.** The registered study failed its advancement gate. The audit below preserves that result and explains its scope. It uses the sealed target scores after prediction lock, and measures upper envelopes of the finite observed utilities on the three reused development families. These are neither bounds on a future query population nor evidence that an oracle policy can be learned or transferred.

The strongest source-fixed baseline happens to select the target-best fixed global policy in every family: original+HyDE for FiQA, keywords+HyDE for NFCorpus and SciFact. Even a target-label oracle choosing a different global policy for every lambda could gain only **0.00230748 macro nDCG**, below the registered **0.005** requirement. The gate was therefore unattainable on these observed utilities for the entire registered global-policy class. This does not justify changing the gate after evaluation.

| Family | Source-fixed160 = target-best fixed global | Target-best per-lambda global | Available global adaptation gain |
|---|---:|---:|---:|
| FiQA | 0.35211561 | 0.35267816 | 0.00056254 |
| NFCorpus | 0.25746707 | 0.26055629 | 0.00308921 |
| SciFact | 0.67890305 | 0.68217372 | 0.00327067 |
| Equal-family mean | 0.42949525 | 0.43180272 | 0.00230748 |

This diagnoses a weak adaptation test for this truncated weighted-RRF operator, uniform seven-point lambda grid, eleven-policy menu, and one-global-policy deployment rule. It does not rule out useful onboarding with other operators, policies, observations, or query-dependent decisions.

The actual within-corpus bridge reaches 0.42898078, below fixed160 by 0.00051446. Source calibration chose eta **0, 1, 1** for FiQA, NFCorpus, and SciFact; the pooled model chose **0, 0.5, 1**. All ten non-reference slopes have nondegenerate source feature variance. Thus FiQA genuinely rejects the correction through source calibration, while the other two folds pay for a correction that changes few final decisions. This is not simply a numerical zero-variance collapse.

The known-lambda/source-utility comparator reaches 0.42586393 and loses to fixed selection in FiQA and SciFact. It has better centered response MSE, 0.00012199 versus the zero-response model's 0.00016837, yet worse final utility. The within model's smaller MSE improvement, to 0.00016318, likewise does not produce a useful policy. Predicting backend-sensitive response curves and preserving the target ordering between actions are separate problems; knowledge of lambda alone does not supply target utility levels.

The saved mean-profile aliases impose little additional capacity loss for global decisions: **0.00017675 macro nDCG**, averaged over the three observed panels. Only two repeated-profile groups have incompatible global optima: NFCorpus seed47 at lambda indices [4,5], and SciFact seed47 at [3,4,5]. The full per-document RR vectors also coincide inside these groups, but the permitted returned ranklists differ. Consequently, these are losses from the RR representation in this finite grid, not indistinguishability of the full permitted observation interface. Distinct saved ranklists do not themselves establish a transferable decoder.

## Query-dependent capacity

The evaluator can choose policies using the already-existing four label-free query buckets, `2*(surface token count>8)+any identifier-like token`. The audit verifies all query IDs against the frozen query-transfer evidence and reconstructs bucket membership from its saved lexical features. It uses only the same eleven policies within the two-search cap; the six-search all-action reference is excluded. Queries retain equal weight within each family. Empty buckets contribute zero, rather than receiving artificial weight.

| Evaluator policy class | Macro nDCG |
|---|---:|
| One global policy fixed across lambda | 0.42949525 |
| One global policy per lambda | 0.43180272 |
| One policy per query bucket, fixed across lambda | 0.43518319 |
| One policy per query bucket and lambda | 0.44156997 |
| One policy per individual query, fixed across lambda | 0.51890877 |
| One policy per individual query and lambda | 0.52861647 |

Bucket conditioning adds **0.00976725** over the per-lambda global oracle. The gains are 0.00847186 for FiQA, 0.00638788 for NFCorpus, and 0.01444200 for SciFact. The per-query oracle adds a much larger 0.09681375. There is meaningful finite-sample policy capacity outside the registered global class.

However, most individual-query capacity exists without adapting to lambda. Allowing backend-specific choices adds only **0.00638677** over the best fixed four-bucket policy, and **0.00970771** over the best fixed per-query policy. The large per-query ceiling chiefly motivates query routing; it does not by itself motivate onboarding. A 0.005 improvement over a truly strong fixed bucket policy would already require most of the observed bucket-class adaptation room.

All these choices maximize and score the same target labels. Small cells make the bucket maxima optimistic as indicators of transferable future gain. Test bucket counts are FiQA [31,20,61,38], NFCorpus [57,4,4,0], and SciFact [14,21,36,79]. This is a capacity audit, with no test-set model selection or deployment of the oracle choices.

## The next research step

The next investment should be additional independent source families and backend interventions with supported query categories. Before another acquisition study, screen this source-only material for useful changes in policy ordering. A subsequent contextual extension should compare a probe-dependent policy against an **equally source-calibrated contextual policy that uses no probes**, retaining global fixed160 as an additional reference. Otherwise a gain could come entirely from the freely observed query category. Keep a shared-slope contextual comparator so that a benefit from query-dependent utility levels can be separated from any benefit of fitting separate response slopes.

Source support argues for explicit count-based shrinkage of bucket utilities and response coefficients toward the corresponding global quantities. There are only two source corpora per fold. Reusing a query's labels over seven lambdas and three document panels does not create independent utility observations.

| Family | Train query counts by bucket | Calibration query counts by bucket |
|---|---|---|
| FiQA | [32,14,52,30] | [7,1,11,13] |
| NFCorpus | [108,14,3,3] | [25,4,2,1] |
| SciFact | [9,15,28,76] | [1,2,7,22] |

A separate source-only exploratory check fits static bucket utility gaps and mixes them with the global gap using eta in {0,0.5,1}. Source calibration selects contextual eta1 in the FiQA-heldout and NFCorpus-heldout folds, but eta0 in the SciFact-heldout fold. The selected source calibration gains, 0.02272 and 0.02206, are entirely driven by SciFact: the other source corpus loses 0.01012 and 0.00899 respectively. The buckets may partly proxy family-specific utility levels. These calibration results do not establish transfer.

The source support and concentration of calibration gains do not justify another target-evaluated model on these same data now. After collecting stronger source evidence, freeze one small partially pooled design and the full calibration/null pipeline; keep the same observation and cost contract. A result that merely beats global fixed160 but fails against the no-probe contextual policy does not establish onboarding value. Evaluate on untouched families. Broadening source variation is more consequential than adding acquisition lookahead to this bridge.

## Reproduction and evidence

`posthoc_intervention_headroom.py` independently reconstructs all policy-class envelopes and profile alias losses from sealed scores and decisions. It checks the freeze and evidence hash links, validates query identity and label-free bucket membership, tests query-weighted aggregation on two analytic examples, and checks the nested-class inequalities. Its output is `posthoc_intervention_headroom.v1.json`.

`posthoc_context_support.py` calculates from existing source train/calibration utilities and query metadata, with no target test utilities entering the calculation. Its monolithic query-evidence input physically also contains prior endpoint test scores, but these fields are never indexed. The output, `posthoc_context_support.v1.json`, records the exact small static-context calibration check described above. It is a diagnostic for a possible design, not a new registration.

Run either script with the existing pilot Python environment. Both refuse to overwrite outputs; pass `--output /tmp/new-audit.json` for a repeat. All source digests and the producing script digest are recorded in the JSON. No registered study file was changed by this audit.
