# Intervention utility transfer: negative result

The registered bridge did not improve retrieval over either source-fixed fusion
baseline. Mean nDCG@10 was **0.428981**, versus **0.429495** for the fixed policy
using the same 160 labeled questions per source family. Each fold has two source
families; a question can have multiple document judgments. Their paired difference
was **−0.000514** (conditional 95% interval **[−0.003511, 0.002125]**). The practical gate
failed; no target family improved over that baseline.

These are equal-family, equal-lambda, equal-panel means over three previously
explored families. Lambda variants and panel runs share query units. The interval
resamples each query's complete lambda/panel vector, with sources and panels fixed.

| Method | Mean nDCG@10 |
|---|---:|
| Centered bridge, source-calibrated strength | 0.428981 |
| Source-fixed, 128 training questions per source family | 0.429201 |
| Source-fixed, 160 training+calibration questions per source family | 0.429495 |
| Pooled bridge | 0.429061 |
| Centered bridge, fixed strength 1 | 0.425276 |
| Source utility with privileged known lambda | 0.425864 |
| Original query | 0.411318 |
| HyDE | 0.375912 |
| RRF-all, higher serving cost | 0.433357 |

| Target family | Centered bridge | Fixed160 | Difference |
|---|---:|---:|---:|
| FiQA | 0.352116 | 0.352116 | 0.000000 |
| NFCorpus | 0.256786 | 0.257467 | −0.000681 |
| SciFact | 0.678041 | 0.678903 | −0.000862 |

The difference from Fixed128 was −0.000220, with interval [−0.001135, 0.000757].
The frozen gate required at least +.005 macro nDCG, positive differences in at
least two families, and no family loss beyond .005, against both fixed baselines.

**The finite benchmark cannot support that gate within this policy class.**
Fixed160 equals the evaluator's best fixed policy in all three target families.
Even an oracle choosing the best global policy separately for each lambda could
gain only **0.0023075** macro nDCG over Fixed160, below the registered .005 threshold.
This is a bound for these test questions and environment-level policy choices,
not a general information or utility-transfer impossibility.

Source calibration selected zero correction for FiQA, so its primary policy paid
no onboarding cost. NFCorpus and SciFact selected full correction and each paid
48 wrapper searches, 96 component searches, 16 generation calls, and 8 document
samples per eight-document panel, with recorded token costs. Fixed policies paid
no onboarding. Primary serving averaged 1.952 wrapper calls on NFCorpus and 2 on
the other families; component calls are twice those counts. RRF-all used 6 wrapper
and 12 component searches plus one shared generation bundle per task, so its
quality advantage is not a comparison at the same serving cap.

The centered model reduced mean centered-response MSE from 0.000168370 to
0.000163179, but that small improvement did not yield better decisions. It also
trailed the pooled model and the best registered shuffled control in retrieval
quality. The fixed-strength diagnostic increased response error to 0.000194769.
These results do not establish useful transfer of backend response.

Giving the comparator the true lambda still did not fix source utility ordering;
that comparator is a diagnostic, not an upper bound. Evaluator-only adaptation
headroom within the 11-policy menu was 0.000563 for FiQA, 0.003089 for NFCorpus, and
0.003271 for SciFact. These interventions offer little room for better global
choices. A more elaborate acquisition policy is premature without a benchmark
that has enough decision headroom; the current result is not a general failure
of backend identification.

The complete numeric output, including all fixed policies, per-policy response
errors, mean-profile aliases, panel variation, and full resource vectors, is in
`results.v1.json`. No target-centered diagnostic fed back into policy decisions.

## Audit and replay

Independent reconstruction found no discrepancies. It checked all 8 frozen
implementation hashes and 13 source hashes, 1,638 decisions, 959 input probe
records, 6,720 paid observations, the weighted Fixed160 prior, costs, response
diagnostics, and paired intervals. Separately, 40,560 endpoint nDCG/recall values
were reconstructed from prior action rankings with an independent metric/fusion
implementation; the largest difference was 3.33e−16. Another 8,450 single-action
scores matched the archived outcomes to the same precision.

The code freeze is commit `16f668d`; the prediction lock is commit `b6b55ef`.
Predictions were saved at 14:23:30.311107 UTC, before new target metric computation
began at 14:23:52.731112 UTC on 2026-09-10. The source provider physically parsed a
monolithic archive containing qrels; the decision stage received no test utilities.

`replay_portable.py` creates a fresh copy containing frozen code and portable
artifacts, with **no raw cache directory**. Regenerated decisions match exactly
after excluding only the new lock-creation timestamp. Evaluation uses the unchanged
copied original lock to preserve its test-label hash binding and produces
**byte-identical results**. Neither lock nor the test-label hashes are rewritten.
Original frozen bytes remain unchanged. From this directory with Python and
NumPy installed (see `README.md`), run:

```sh
python replay_portable.py
```

Audit records: `independent_audit.v1.json`, `endpoint_metric_audit.v1.json`, and
`portable_replay_audit.v1.json`. The replay helper and audit notes were added after
the run; the registered code, protocol, and result artifacts were preserved.
