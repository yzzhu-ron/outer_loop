# SearchProbe: from an unresponsive router to a measurable intervention

**Completed 10 September 2026.** Semantic actions substantially increase the
available per-query headroom. The frozen probe selector does not exploit it:
all four selectors leave every action unchanged. A label-free sensitivity
certificate explains this result: under the supplied score bounds, no allowed
history of at most 32 probes can overcome the fitted router's score margins on
any evaluated query.

This is evidence about a specific adaptation pipeline. It does not show that
the search observations lack useful information. The
[research report](../searchprobe-research-report.html) connects this result to
the follow-up, theory and [SearchProbe](../../../packages/searchprobe/) tool.

## Completed experiment

The main protocol was committed in `c08f62d`; diagnostics and report validation
were committed in `f696c36` before semantic retrieval. The original
[lexical pilot](../pilot/RESULTS.md) remains unchanged. The original and keyword
actions reproduce its per-query nDCG exactly on all four shared configurations:
1,200 action/query comparisons, zero differences.

The local Qwen generator produced 142 question pairs, 284 probe rewrites and
941 real-query rewrite responses. One NFCorpus calibration response was
malformed; all 365 test-query rewrites were format-valid. Failure did not cause
retry, deletion or fallback. Format validity does not establish faithfulness;
the [fixed text audit](GENERATION_AUDIT.md) documents changed constraints and
unsupported premises without consulting retrieval outcomes.

Each corpus has 192 source queries partitioned 128/32/32 for router fitting,
calibration and selector utility. Each fold holds out one whole corpus and uses
the other two at the same backend. Three onboarding seeds provide 16 sampled
documents and 128 candidate pairs each. Four selectors use budgets 0/4/8/16/32.

| Configuration | Queries | Original | Source router | Best fixed oracle | Per-query oracle | Oracle − fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FiQA / BM25 | 150 | .2525 | .2525 | .2525 | .3692 | .1167 |
| FiQA / dense | 150 | .3921 | .3325 | .3921 | .5303 | .1381 |
| NFCorpus / BM25 | 65 | .2451 | .2451 | .2485 | .2990 | .0505 |
| NFCorpus / dense | 65 | .2348 | .2348 | .2348 | .2968 | .0620 |
| SciFact / BM25 | 150 | .6061 | .6061 | .6209 | .7705 | .1496 |
| SciFact / dense | 150 | .6436 | .6436 | .6795 | .7892 | .1097 |

Values are nDCG@10. Both oracles use target labels and are evaluator-only.
The five actions are original, keywords, semantic rewrite, hypothetical passage
and two-query decomposition. A bad average action can still improve some
queries; the oracle exploits these differences with unavailable labels.

## Why the original selector cannot improve

Source calibration chooses the fixed-score baseline over either query-feature
Ridge model in all six folds. The profile correction then has insufficient
amplitude to change those baseline decisions. If an action's correction has
coefficient vector c, its absolute movement is bounded by
`||c||_1 × B/(B+4)` after B probes. Comparing winner/challenger bounds with the
base margin certifies **100% unchanged queries at B=32 in every fold under the
supplied score/bound model**.

There are 3,480 one-step source-utility training rows per fold, 20,880 in total;
every reward is zero. These reuse only 64 utility queries per fold. All 450
method/seed/budget records preserve their baseline decisions. Query/seed
bootstrap intervals of [0,0] describe this exact equality, not certainty about
unseen corpora or a universal absence of exploration value.

The optional full-pool diagnostic also changes no actions after observing all
128 pairs. That aggregate is not a selector oracle: useful subsets can cancel
when aggregated. The margin certificate supplies the stronger conclusion for
the allowed 32-pair budget independently of target utility labels.

## A responsive follow-up

The source-world intervention was designed after that result, then committed
in `64cab28` before its own target evaluation. It reuses the original tasks,
actions, candidate pools and costs, preserving every original output. It uses
only the two other corpora's 128-query source-training partitions to fit
bucket-specific action utilities, with each source corpus a possible world.
Selected probes update a posterior over those worlds. Random, fixed,
information gain and decision value share this source model and prior.

This intervention changes decisions in every environment. At 32 pairs,
decision-value selection produces:

| Environment | Own prior | After probes | Change from prior [95% interval] | Change from random [95% interval] | Actions changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| FiQA / BM25 | .2398 | .2313 | −.0085 [−.0267, .0082] | −.0028 [−.0135, .0031] | 79.3% |
| FiQA / dense | .3250 | .3325 | +.0075 [−.0288, .0449] | .0000 [.0000, .0000] | 40.7% |
| NFCorpus / BM25 | .2451 | .2455 | +.0004 [−.0046, .0040] | −.0005 [−.0049, .0025] | 35.4% |
| NFCorpus / dense | .2258 | .2356 | +.0098 [−.0175, .0356] | +.0078 [−.0263, .0471] | 35.4% |
| SciFact / BM25 | .6230 | .6112 | −.0118 [−.0385, .0144] | .0000 [.0000, .0000] | 47.3% |
| SciFact / dense | .6376 | .6795 | +.0418 [−.0056, .0922] | +.0094 [−.0005, .0274] | 100.0% |

The strongest descriptive result is SciFact/dense. All three runs select HyDE
for every test query, reaching the target-best fixed action. This improves over
both the new prior (.6376) and the original frozen router (.6436), whose paired
change is +.0359 [−.0134, .0864]. It does not capture the per-query oracle's
.7892 quality. The primary own-prior comparison distinguishes probe-induced
adaptation from changing the starting model. Every reported interval includes
zero; these are promising exploratory measurements, not a confirmed transfer
or acquisition-method improvement. The report shows all four methods and all
five budgets, without selecting a better budget using target labels.

Information gain and decision value have identical quality at budget 32 in all
six environments, although their selected histories differ. With two worlds
and shared Gaussian variance, equal-cost channels are Blackwell ordered by
standardized mean separation. Exact information gain and decision value prefer
the same most informative equal-cost channel; costs, ties and quadrature can
produce differences here. This experiment therefore tests responsiveness but
cannot demonstrate a distinct benefit from ignoring nuisance-world information.

The likelihood is an uncalibrated Gaussian surrogate. Repeated probes share
documents and queries, so their conditional independence assumption is false.
The posterior describes source-model weights, not verified confidence about
the target. Evidence from long-query probe buckets changes short-query decisions
through a global-world assumption, an extrapolation especially consequential
for NFCorpus. Source worlds need not cover target variation.

Every positive budget still pays full pool generation and inspection. All 360
run rows and 1,440 cost rows, the three reference policies, every posterior
trace, fitted source-model provenance and paired outcomes are saved in
[world_model_followup/](results/world_model_followup/). No additional model
generation or corpus retrieval was required; cached outcomes retain their
logical deployment costs.

## A reusable responsiveness diagnostic

`searchprobe response-audit` turns the original bottleneck into a general
preflight check. Given base scores and justified absolute correction bounds,
it exactly computes which rows cannot change, honors action tie order, and
lists feasible alternative winners in the declared score box. It also bounds
the absolute mean change of any fixed utility in [0,1] by the potentially
changed fraction. A change may help or hurt; the tool uses no relevance labels.

The [six real score models](results/response_models/) reproduce 100% unchanged
at budget 32 with the same public API and CLI as the synthetic examples.
Arithmetic is exact for the supplied decimal/rational model. The exported
scores and bounds originate in floating-point calculations, so this is not
formal verification of the entire running implementation. Proving bound
validity and transfer remains the caller's responsibility. The interval
comparison is elementary robust argmax analysis, not a new theorem.

## What the probes do measure

Between 7.8% and 42.4% of deduplicated candidate pairs have nonzero known-document
RR contrasts, depending on corpus/backend. The probe outcomes are not all
constant, despite the router's constant decisions.

The independent assistant-conducted trace audit reconstructs every one of
2,304 raw query/action/rank records. The strict single-query export retains
1,728 pairs and excludes 576 decomposition pairs. It has no malformed records,
duplicate IDs, exact/token-bag query collisions or accounting violations.
Of retained pairs, 1,360 have equal target ranks, 1,195 rank the target first on
both sides, and 123 miss it on both sides. These are known-item endpoints, not
complete relevance judgments or equality of result lists.

A post-result support audit found that every probe occupies a long-query bucket
(more than eight tokens). Short queries comprise 35/150 SciFact, 51/150 FiQA and
61/65 NFCorpus test queries. Default tool coverage checks cross observed buckets;
declaring expected buckets 0/1/2/3 is necessary to flag the missing short-query
cells. This matters especially for NFCorpus. The saved
[independent trace review](results/independent_trace_review.json) gives counts
and provenance. It is a technical cross-check by another assistant, not external
human validation.

## Comparator and costs

Source-selected original-plus-alternative RRF changes nDCG relative to the source
router by +.0018, +.0566, +.0034, +.0030, +.0043 and −.0022 in the table's order.
The largest gain repairs a poor transferred FiQA/dense HyDE choice: RRF .3891
still trails the original query's .3921. This is not evidence that RRF improves
over the original query everywhere.

At 32 pairs, learned selection pays 16 document inspections, 48 setup LLM calls,
about 5,264–5,948 setup output tokens and 71.7 probe searches averaged across
seeds, with zero quality change. Other selectors have their own exact search
counts. Serving costs remain additional. The
[cost curves](results/cost_curves.csv) retain all five resource categories and
workloads of 1/10/50/200 future tasks. No dollar or latency dominance is claimed.

## Scope and evidence

NFCorpus retains 65/323 test queries under the all-positive-support holdout rule;
the retained distribution is unusually restricted. The three corpus families
and two correlated backends are exploratory. Query/seed intervals condition on
this panel, with no multiplicity adjustment or population-transfer guarantee.
Dense input truncation is 256 tokens. The task is retrieval routing, without an
end-to-end answer-correctness evaluation.

Exact source-model decision certificates are also saved for all six folds,
using only the two source-training utility vectors. They concern best fixed
actions in the declared source models. The resulting radii range from 0 to
.0169; their small values cannot explain or rule out the larger per-query oracle
gaps because they have different policy menus and information contracts.

The useful methodological finding is that headroom, probe contrast, decision
responsiveness and transfer are separate empirical questions. This experiment
isolates a responsiveness failure, making a direct intervention possible.
