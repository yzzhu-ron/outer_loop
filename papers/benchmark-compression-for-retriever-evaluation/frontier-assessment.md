# Assessing the “Compression–Reliability Frontier”: Insufficient Novelty in the Current Proposal

Research assessment · 8 September 2026

Related documents: [original research proposal](README.md) ·
[alternative research directions and ratings](research-directions.md).

Assessment of [outer_loop PR #1](https://github.com/yzzhu-ron/outer_loop/pull/1), its earlier RSI benchmark, and the primary literature.

**My recommendation is to refine the research claim and run a small pilot before drafting the paper.** Reliable cheap evaluation is a valuable problem. The move away from making RSI the headline is sensible. But the proposed frontier, inverse budget, and much of the suggested methodology already have close precedents. Executing the current outline faithfully would not, by itself, give me confidence in an ICLR submission.

A convincing paper needs a specific advance in decision reliability, transfer, or cost that survives the strongest existing baselines. A useful possible direction is evaluation that can certify a comparison or request more evidence. That direction also has substantial prior work; it is a research hypothesis to test.

Reviewed PR head `ffeb67e84a611d2876a7dd0e3b1b425f0528e254`, merged September 8, 2026. The review covers its full 1,331-line plan, the pre-PR version, the earlier benchmark release and baseline report, and the surrounding theory/ShopGym records. No new paper experiments were run for this assessment. References to the current plan below refer to that reviewed revision.

<a id="context"></a>

## What changed, and what the existing evidence actually says

The broader `outer_loop` program studies improving prompts, programs, harnesses, and evaluation sets around a frozen model. Its shared package contains selection and archive primitives. The general manuscript develops theory about proposals, feedback, noisy selection, and generalization. The ShopGym application has software and simulation results; its README explicitly says live empirical evidence is still needed. These are related research tracks, rather than experimental support for this new compression paper.

| Stage | Question and evidence |
| --- | --- |
| Original agent benchmark | Can an agent improve a deterministic query selector? The release has 40,159 queries across 13 datasets, six visible retriever families, three sealed families, and nested budgets of 650 / 1,300 / 3,250. It uses Liquid NanoBEIR-seeded, relevance-complete *reduced corpora*, uniform query weights, and a task-specific composite fidelity reward. |
| Pre-PR paper idea | Compare random edits, structured-feedback search, and archive search on the selector. The proposed contribution was search efficiency and transfer, with the earlier visible-to-hidden gap as motivation. |
| Merged PR | Study the smallest retained query budget supporting a declared decision-reliability target. AutoCompress-IR would learn weighted, nested policies; the paper would map curves across six full-corpus BEIR datasets and roughly 16 development plus four frozen systems. RSI is now an optional optimizer. These experiments and guarantees are proposed. |

The frozen mean-matching selector scored **0.979940 visible versus 0.874672 hidden** on the earlier task. Those numbers are composite rewards combining pair order, score accuracy, winner retention, dataset and macro aggregation, and three budgets. They are not 97.99% versus 87.47% probabilities of correct pairwise decisions under the new protocol.

There is an additional useful diagnostic: in the earlier leave-one-visible-family-out study, equal-allocation random sampling scored **0.900280** and visible mean matching **0.898666**. This is a different diagnostic from the frozen hidden score, and it does not establish a statistically significant difference. It does show why excellent fitted performance should not be treated as evidence that an optimized selector will transfer.

The old Anchor Points and tinyBenchmarks implementations were adaptations constrained by the task’s uniform-weight, nesting, and allocation contract. The new proposal allows learned weights and a different objective. Those old baseline scores cannot establish superiority over the published methods under the new rules.

**What is reusable:** outcome-matrix tooling, selector interfaces, provenance practices, and inexpensive exploratory diagnostics. **What needs new evidence:** full-corpus retrieval behavior, the broader system panel, an actual AutoCompress algorithm, calibrated budget selection, and measured savings. The exposed old hidden result cannot serve as a fresh confirmatory test.

<a id="literature"></a>

## The closest literature changes the novelty assessment

The broad idea has a long history in IR and a recent wave in LLM evaluation. The following comparisons are grounded in primary methods, experimental sections, and targeted appendices. Results reported by those papers were not independently reproduced.

| Work | What it already establishes | What remains different here |
| --- | --- | --- |
| [A Few Good Topics](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/FewerTopics.pdf)<br>Guiver, Mizzaro & Robertson, TOIS 2009 | Curves over topic counts; best/average/worst subsets; inversion at a desired correlation; pairwise and rank-based criteria; held-out-system evaluation. Section 3’s example reaches .95 *linear score correlation* with six best-selected, 22 random, or 41 worst-selected topics. Sections 4.5.1 and 5.6.1 discuss held-out systems and correlated submissions. | The .95 example is not .95 pairwise agreement. Judging cost motivates much of the work, but §6.2 also proposes condensed official topic sets for repeated system optimization and full validation. The PR needs more than the frontier framing. |
| [Uncertainty-aware query selection for evaluation](http://www0.cs.ucl.ac.uk/staff/ingemar/Content/papers/2012/SIGIR2012.pdf)<br>Hosseini et al., SIGIR 2012 | Section 7.1 gives required query counts at target Kendall correlations: at τ=.9, random needs 739 and Adaptive 486 in the reported setting. Section 8.1 holds out participant groups, finds transfer failure for Adaptive, then develops Adaptive+ using voting over system subsets. | Its uncertainty concerns relevance/performance estimation under a judging protocol. It does not supply the proposed finite-sample confidence guarantee for selecting a query-inference budget for future retrievers. Still, target-size selection and panel-transfer remedies are established. |
| [Intelligent topic selection for low-cost IR evaluation](https://www.ischool.utexas.edu/~ml/papers/kutlu-ipm18.pdf)<br>Kutlu, Elsayed & Lease, IP&M 2018 | Greedy Kendall-preserving selection, learned incremental selection, and detailed budget–reliability analyses. Algorithm 1’s greedy method is called an oracle because their task lacks qrels before selection. | With the PR’s permitted full development-system outcomes, that greedy method is an ordinary eligible baseline. Its judging-cost analysis differs from query inference; its pooling-reusability study is not a clean system holdout from selection. |
| [How Reliable is Language Model Micro-Benchmarking?](https://arxiv.org/html/2510.08730v2)<br>Yauney et al., ICLR 2026 | Defines held-out pairwise ordering agreement conditioned on full-benchmark margins; introduces Minimum Detectable Ability Difference; sweeps sizes and retained proportions; compares source/target panels and held-out examples. §§3–5 directly analyze evaluation efficiency versus reliability. | Primarily a study of existing compressors. A better decision-focused method or a useful risk guarantee could extend it. “We examine decision reliability rather than score error” is already its main motivation. |
| [Benchmarking on Tasks That Matter](https://arxiv.org/html/2606.27997v1)<br>Gusev & Zaytsev, KDD 2026 | Ranking-preservation curves, farthest-first nested selection, margin/variance theory, and cross-domain differences. §6.2 defines the smallest `k*` meeting a target, both on a mean curve and using the conservative side of its uncertainty interval. | Selects whole datasets rather than queries, and its empirical-quantile intervals are not a demonstrated simultaneous risk certificate for adaptive method/budget selection. The inverse frontier and conservative threshold concept nevertheless have direct precedents. |
| [How benchmark prediction from fewer data misses the mark](https://arxiv.org/html/2506.07673)<br>Zhang, Dorner & Hardt, NeurIPS 2025 | Eleven methods on 19 benchmarks, each with at least 83 models. Random sampling plus regression is strongest in interpolation; performance often deteriorates under extrapolation. §3 uses reference-model outcomes to predict target outcomes and corrects the estimate with AIPW. | This is a crucial missing citation. It already challenges the importance of clever subset selection and connects transfer to behavioral similarity. A learned surrogate plus random residual correction alone is also insufficient novelty. |
| [Anchor Points](https://arxiv.org/html/2309.08638v2)<br>EACL 2024 | Correlated outcome representations, weighted medoids, learned predictors, held-out models, budget curves, low-rank behavior, and family-transfer experiments. | Leaves source-panel design and rigorous generalization guidance unresolved. Include the weighted/predictive variants, with clearly matched access to information. |
| [tinyBenchmarks](https://arxiv.org/html/2402.14992v2)<br>ICML 2024 | IRT-based selection and estimation, learned weights, held-out and temporal model splits, specialized models, size sweeps, and bounded continuous scores (§4.3). | Focuses on score estimation. Compare against appropriate IRT++ / prediction variants, rather than treating a constrained clustering adaptation as the whole method. |
| [EssenceBench](https://arxiv.org/html/2510.10457)<br>2025 preprint | Iterative genetic subset search with attribution-guided refinement, full-score prediction, budget sweeps, and ranking diagnostics. | Reinforces that an outer search loop is familiar. Its “95% ranking preservation” concerns rank displacement, not 95% correct pairs; those numbers are not directly comparable. An executable public implementation was not verified. |
| [Coresets Before Score Sets](https://arxiv.org/abs/2607.09739)<br>2026 preprint | Held-out-model evaluation on 18 models and 35 benchmarks; budget sweeps; semantic facility-location selection that performs well when a small model panel makes IRT unstable. | Small-panel instability is directly relevant to 16 development systems. Semantic facility location is an important baseline. |

[Active Sampling for Large-scale Information Retrieval Evaluation](https://arxiv.org/abs/1709.01709) (CIKM 2017) additionally compares ranking quality across judging budgets and uses leave-one-group-out evaluation. Its document/relevance-judging cost model should be distinguished from reducing query execution. The older IR methods should inform both the history and the baseline design; methods that require unavailable judgments or a different cost model need an explicit adaptation.

**The introduction needs revision.** The PR’s assertion that existing methods do not answer how much compression is safe is too broad. A more accurate starting point is: existing methods and studies characterize size–fidelity tradeoffs; the unresolved problem this paper must identify is a particular form of reliable deployment that those methods cannot deliver efficiently.

The frontier remains a useful organizing figure. Taking a maximum over methods and inverting a curve does not create a new scientific result. Similarly, with no ties, pairwise agreement is exactly `(1 + Kendall τₐ) / 2`.

<a id="design"></a>

## Changes needed before expensive experiments

### Specify the decision that matters

Average pairwise agreement is useful, but “95% of pairs agree” does not mean “the correct winner is selected with probability 95%.” Swapping the top two of 20 systems preserves **189/190 = 99.47%** of comparisons while choosing the wrong winner.

Declare a practical target: a candidate versus a fixed incumbent, choosing among competitive systems, or estimating all scores. Put close-margin error and winner/selection regret alongside aggregate agreement. Define practical ties explicitly. Comparisons among very weak and very strong models should not dominate the claim about selecting between competitive systems.

### Resolve what the confidence statement is about

The proposed 16-reference/four-held-out panel yields 64 held-out–reference pairs and six held-out–held-out pairs. They involve four fresh system units, not 70 independent future systems. Also, 91.4% of the headline pair mixture includes a system used during construction. That can be a valid use case, but it differs from comparing two unseen systems.

A curated panel plus repeated folds can measure empirical transfer within the chosen panel. It does not automatically certify 95% reliability for a population of future retrievers. Simultaneous confidence bands address selection over budgets only after the underlying inference is valid. They do not solve small sample size, overlapping folds, adaptive policy tuning, or an undefined sampling population.

<details>
<summary>Why four perfect systems cannot establish a distribution-free 95% population certificate</summary>

Consider one frozen policy and one budget. Even if every observed fresh system has perfect fidelity, an unrestricted population may contain an unseen bad-system subpopulation. In the extremal Bernoulli example, the one-sided 95% lower bound after *n* independent perfect observations is `0.05^(1/n)`: .473 for four systems, .829 for 16, and .9505 for 59. This is an illustration of the information limit, not a proposal to apply binomial intervals to dependent model pairs. Structural assumptions or a different source of randomization can change the problem. Full evaluation has deterministic equality and is a special exception.

A bootstrap can give a falsely reassuring [1,1] interval when the observed panel has no failures. Coverage simulations must include rare failure types and correlated systems. One final four-system test is a valuable non-adaptive check, but cannot itself establish a calibration procedure’s long-run coverage.

</details>

Choose among three honest claims: empirical panel-transfer performance; population reliability under an explicit, supportable sampling/structural assumption; or query-randomization confidence for a particular frozen comparison. Settle this before committing to the abstract’s “smallest reliable budget” promise.

### Make AutoCompress-IR a specific, testable method

Frank–Wolfe, sparse weighting, rounding, swaps, robust losses, and cross-validation are an implementation menu. Specify the exact objective, fitting procedure, and mechanism expected to improve held-out decisions. Start with one hypothesis and a strong baseline. Adding more optimizer components is not evidence of a research contribution.

Pair-difference vectors contain useful structure, but their dimension is at most *K−1*, despite having *K(K−1)/2* coordinates. Moreover, `Σᵢ<ⱼ(eᵢ−eⱼ)² = K Σᵢ(eᵢ−mean(e))²`. Ordinary squared pair reconstruction is centered score reconstruction; a genuinely different objective must come from decisions, margins, costs, or transfer assumptions.

Promote the closest eligible IR and modern methods into the main comparison. Include random plus regression and AIPW from Zhang et al., semantic facility location, a strong greedy decision baseline, and properly implemented Anchor Points / tinyBenchmarks variants. Where estimators have more freedom than weighted means, report the restriction and include a broader practical comparison.

### Control for simple explanations of benchmark differences

TREC-COVID has 50 queries; Quora has 10,000. If both needed the same absolute number of queries, their required fractions could already differ by 200×. A twofold variation in retained percentage is therefore a weak discovery threshold.

Report absolute required query counts as well as percentages; match decision margins; use matched-size subsets or other controls for benchmark size. Avoid fitting a many-factor “law of compressibility” from six dataset averages. The stronger result would predict error or required cost on held-out conditions beyond simple count, margin, and variance baselines.

A common safe ratio can always be chosen as the largest required ratio, including 100%. Differing minima show that a fixed ratio can be inefficient; they do not prove that no universal safe ratio exists.

### Use theory that matches the information available

T1’s Carathéodory result is correct and classical. T2’s bounded random-sampling rate is standard and may be numerically loose. Treat these as foundations unless a substantial new consequence follows.

T3 needs an explicit observation model. A variance-over-squared-margin bound for learning an unknown pair from sampled outcomes is not a lower bound on the size of an optimal weighted subset selected after seeing all outcomes. For one known pair with a positive mean, one positive-difference query preserves its sign; at most two queries can reproduce its exact mean. The small-instance optimization oracle and the unknown-system sampling bound answer different questions.

The most consequential theory would justify the actual decision rule and budget selection, or predict transfer in a way that guides an improved method. A paper does not need every proposed theorem, a new optimizer, six datasets, and every artifact contribution to be strong. It needs a clear central advance with sufficient evidence.

<details>
<summary>Additional formulation and implementation corrections</summary>

- Use a cost-at-most budget for a monotone optimal frontier. An individual method’s measured reliability can decrease when its support or weights change.
- At the 100% endpoint, restore the original benchmark weights. Including every query with arbitrary learned weights does not reproduce the full mean.
- System-dependent query costs need a declared reference, average, or worst-case cost function for one common frontier. Otherwise use query count on the horizontal axis and report realized workload costs separately.
- Report actual integer support. One TREC-COVID query already costs 2% of its query count; nominal 1% and 2% points may collapse.
- Call the result the smallest *tested and supported* budget. Add measurements near threshold crossings before claiming a 1.5× reduction from a coarse grid.
- Keep the optimizer’s validation folds separate from outer evaluation. Optimizing against a “held-out-development fold” makes it development data.
- Distinguish reliability averaged over training procedures from a guarantee for the one released, refitted policy.
- If each dataset has independent weights and budgets, explain what actually couples the worst-dataset objective across datasets.
- Account for building the reference run panel and calibrating the policy. If construction costs *C* and each future evaluation saves *s*, break-even requires roughly *C/s* uses, in addition to fixed-index accounting.

</details>

<a id="refinement"></a>

## The direction I would test first

**Can reference-model behavior reduce the cost of making a correct retrieval comparison, while the evaluator can detect insufficient evidence and escalate?**

This keeps the valuable scientific motivation and makes the success condition concrete. A method could output “candidate wins,” “incumbent wins,” “practical tie,” or “more evaluation required.” Its main figure would compare **query cost at matched error control**, with close decisions and actual selection outcomes visible. The frontier becomes a summary of that capability.

One candidate design is a learned query policy or outcome predictor followed by probability-sampled audits. Fix the retriever pair before auditing. Use fresh randomized evidence to correct prediction error and construct a valid, time-uniform finite-population confidence sequence, or a rule with explicit error spending; if the evidence is inconclusive, evaluate more queries. Repeatedly checking ordinary pointwise intervals is insufficient. Learned audit allocation needs known positive inclusion probabilities and appropriate weighting. If a predictor is fitted using target outcomes, separate or properly account for the data used to fit it. At full evaluation, recover the exact benchmark comparison.

**This design is not yet a novelty claim.** [Prediction-Powered Inference](https://arxiv.org/html/2301.09633v4) already combines predictions and observed residuals. Zhang et al. apply that principle directly to benchmark prediction. [Confidence sequences for sampling without replacement](https://arxiv.org/html/2006.04347v4) already support valid stopping on a finite population (§§3–4). [Fogliato et al.](https://arxiv.org/html/2406.07320) already derive finite-population residual-variance gains for model-assisted evaluation. [Active Testing](https://arxiv.org/html/2103.05331) and [On Speeding Up Language Model Evaluation](https://arxiv.org/html/2407.06172) supply adaptive sampling and best-model-selection precedents. A credible new method must beat a carefully implemented combination of these ingredients, or establish a new result they do not provide.

The research opportunity might be cost-aware allocation that exploits paired retrieval behavior, a useful guarantee under realistic small reference panels, or a measured account of when learned compression helps after the cost of honest auditing is included. Merely adding a confidence sequence to AIPW would be a weak contribution.

An alternative with a stronger connection to the original RSI motivation is **compressed evaluation under adaptive reuse**: after a development loop repeatedly sees the cheap evaluator, does it still choose the same system as full evaluation? A convincing result would measure full-benchmark selection regret at matched total compute and provide a remedy. Random held-out public systems do not test that interaction. This would be a different study, so I would pursue it only if the small pilot reveals a substantial effect and a separate novelty audit supports it.

I would keep lineage/mechanism holdouts as useful sensitivity checks. They need not be harder, and a failure under family shift should not be required to rescue the story. The scientific claim should survive a null result there.

<a id="next"></a>

## The smallest useful next output

1. **A one-page claim and protocol.** Choose the comparison, tie rule, source of randomness, information available to the selector, and exact guarantee. Replace the current broad novelty paragraph using the literature matrix above.
2. **A statistical feasibility check.** Before inference jobs, simulate the intended selection and calibration procedure, including rare failures, close margins, and correlated systems. Determine whether the proposed confidence promise is attainable. A simulation validates behavior in its simulated regimes; it is not a substitute for a theorem.
3. **A small real-data pilot from existing runs.** Use two or three datasets and roughly 10–16 diverse public systems whose full query outcomes are available. Preserve genuinely untouched systems. Old reduced-corpus data can support exploratory debugging, clearly labeled. Do not require the full six-dataset sweep to discover whether the basic idea works.
4. **Strong baselines and decision-focused outputs.** For a static selector, compare fully specified selection/estimation methods under honest outer splits. For an auditing method, compare uniform sequential evaluation and reference-predictor/AIPW baselines at the same error target. Report cost, close-pair errors, winner/regret, absolute query counts, and original aggregate agreement.
5. **A research decision.** Scale only if there is a robust advantage over the strongest relevant baseline, or a new explanatory result that changes how evaluation should be done. Decide which single result will lead the paper. If the result is mostly BEIR curves plus classical bounds, adjust the contribution and venue expectations.

The desirable pilot artifact is a few decisive plots and a short result ledger. A polished abstract and a named algorithm are premature until those plots support a claim.

<a id="iclr"></a>

## My ICLR assessment

| Question | Assessment |
| --- | --- |
| Is the problem worth pursuing? | Yes. Reliable evaluation savings matter for repeated retrieval development and reranking workloads. This can stand independently of a particular RSI framework. |
| Does the current outline establish a distinctive contribution? | No. The related work occupies much of the stated novelty, the algorithm is underspecified, and the central confidence promise is unresolved. |
| Could a refined project reach ICLR quality? | Yes, with a substantive method, predictive theory, or strong empirical discovery supported by correct inference and meaningful baselines. The venue fit is plausible; the needed contribution is not demonstrated yet. |
| Would I expect the outlined study alone to be enough? | I would not. Six BEIR curves, an inverse-size definition, and classical bounds would likely face an incremental-contribution objection. This is a judgment about the proposed contribution, not an acceptance forecast based on nonexistent results. |
| Is ICLR 2027 realistic? | High risk on the documented state. The [official call](https://iclr.cc/Conferences/2027/CallForPapers) lists September 18, 2026 for abstracts and September 25 for papers, both 23:59 Anywhere on Earth. As of September 8, that leaves about 10 and 17 calendar days. A convincing submission would require an already accessible run matrix and a sharp early result; the entire proposed agenda is too broad to assume it will fit. |

I would invest in the pilot, and let its strongest result determine the paper. I would avoid committing to the current title, novelty narrative, or conference deadline before that decision.

<a id="sources"></a>

## Source scope and follow-up reading

The decisive novelty findings above come from primary full texts, including the two older IR papers, the five modern microbenchmark papers, Coresets Before Score Sets, Active Sampling, and the additional benchmark-prediction paper. Prediction-Powered Inference and the without-replacement confidence-sequence paper were checked for the specific proposed refinement. The key citation matrix is sufficient to reject the broad novelty claims; it is not an exhaustive search of all work through September 2026.

- [Merged plan](https://github.com/yzzhu-ron/outer_loop/blob/ffeb67e84a611d2876a7dd0e3b1b425f0528e254/papers/benchmark-compression-for-retriever-evaluation/README.md): definitions lines 188–281; method 472–549; theory 563–707; system design 742–822; baselines 844–870; novelty 1010–1048; gates 1216–1290.
- [Pre-PR paper direction](https://github.com/yzzhu-ron/outer_loop/blob/5522839c2553c7ac667d6d092ed285dff4f88798/papers/benchmark-compression-for-retriever-evaluation/README.md); [earlier benchmark release](https://github.com/yzhu319/agentic_research_benchmarks/tree/main/benchmarks/benchmark-compression-optimization). The earlier figures in this memo were read from the local frozen proposal and baseline-summary records.
- [BEIR](https://arxiv.org/abs/2104.08663): benchmark scope and full-corpus evaluation context. Exact dataset/model versions, query handling, and licenses still need verification when constructing the new study.
- [Kutlu et al., IP&M 2018](https://www.ischool.utexas.edu/~ml/papers/kutlu-ipm18.pdf): §§4–5, supervised greedy selection, learned selection, judging costs and pooling reusability. Published online in 2017.
- [Zhang et al., NeurIPS 2025](https://arxiv.org/html/2506.07673): especially §§3–4, random-plus-regression, AIPW, extrapolation and model similarity.
- [Waudby-Smith & Ramdas](https://arxiv.org/html/2006.04347v4): finite-population observation model, §§3–4 confidence sequences and stopping.
- [Angelopoulos et al.](https://arxiv.org/html/2301.09633v4): §1.3 prediction-powered mean estimation and residual correction.

All eleven papers linked in the PR’s primary related-work list were accessible. The three historical IR papers were read as PDFs, with key tables/pages visually checked. Modern compression papers and Active Sampling were read in full-text HTML; BEIR was read selectively for protocol, efficiency and judgment bias. Additional relevant papers were checked for methods and guarantees. We did not audit every appendix proof or verify availability of every promised code/data release.

This memo separates repository facts, published findings, mathematical observations, and proposed research. It does not certify the cited papers’ proofs, reproduce their experiments, or claim the suggested refinement is novel.
