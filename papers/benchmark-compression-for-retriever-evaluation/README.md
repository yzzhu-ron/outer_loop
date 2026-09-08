# The Compression–Reliability Frontier: How Compressible Is a Retrieval Benchmark?

Status: research handoff and execution plan for a possible ICLR 2027 submission.
No paper experiments described below have been run unless explicitly marked as
pilot evidence.

## Executive decision

Proceed with this direction, but make the submission conditional on an early
go/no-go experiment. The paper should not sell one more clever way to choose a
small query set. Its central move is to replace the question

> What is the best tiny version of this benchmark?

with

> What is the smallest version that reaches a stated reliability target for a
> stated population of retrievers?

The memorable scientific question is:

> **How small can a retrieval benchmark become before it stops reaching the
> same conclusions as the full benchmark?**

The answer is a **compression–reliability frontier**. Its horizontal axis is
the fraction of variable evaluation cost retained. Its vertical axis is the
probability that the compressed benchmark preserves the full benchmark's
model-comparison decisions on retrievers that were not used to construct it.

The core thesis is:

> **Compressibility is not an intrinsic scalar property of a benchmark. It is
> conditional on the benchmark, the target retriever population, the decision
> being preserved, and the evaluation budget. There is no universally correct
> “tiny BEIR.”**

This is a substantially stronger and broader paper than the earlier narrow
proposal. Outer-loop or recursive search may appear as a solver ablation, but
it is not the subject of the paper.

The idea is strong enough to justify the pilot. Full ICLR 2027 execution is
high risk unless the run matrix can be assembled quickly or retrieval jobs can
be parallelized; the pilot must come before a costly full-system sweep.

## The 30-second pitch

Retrieval benchmarks are expensive to run repeatedly, so researchers often
want a small set of representative queries. Existing methods generally return
one subset and judge it on a fixed panel of available systems. A subset can
match those systems extremely well while failing on a new model lineage or a
different retrieval mechanism.

We will first measure whether that failure is real, material, and widespread;
we will not assume it. We then formulate benchmark compression as a
cost-constrained, out-of-retriever generalization problem. We estimate the
best attainable reliability at each budget, introduce **AutoCompress-IR** to
produce nested weighted query policies, and connect transfer to the behavioral
coverage of the retrievers used during construction.

The intended outcome is both scientific and practical:

- a frontier that says how much evaluation cost is actually safe to remove;
- an algorithm that creates a benchmark policy for a declared use case;
- limits explaining why perfect fit on historical systems need not transfer;
- canonical run files that let others reproduce the paper without rerunning
  every retriever.

## Why this matters

A microbenchmark is often used to choose a winner among checkpoints, model
families, or retrieval pipelines. If it was optimized on yesterday's systems,
its errors are adaptive rather than random: it may preserve exactly the
differences represented in its construction panel and miss new differences.
A wrong model-selection decision is more consequential than a small average
score error.

Family-shift failure is plausible, but it is not yet a fact this paper may cite
without evidence. Establishing its prevalence and severity is therefore a
first-class contribution:

1. How often does a subset fitted to known systems lose fidelity on held-out
   checkpoints, lineages, mechanisms, pipelines, or later-released models?
2. Is the loss larger than ordinary variation from selecting fewer queries?
3. At what budgets does the problem disappear?
4. Does broader construction-panel coverage buy reliability, and how much does
   it cost?

If the answer is that random subsets transfer almost perfectly at useful
budgets, that is an important pilot result—but it weakens the case for this
particular ICLR paper.

## Scope

### What is being compressed

The main object is the **query dimension of ranked-retrieval evaluation**.
For a dataset \(B=(Q,C,J)\):

- \(Q\) is the query set;
- \(C\) is the document collection;
- \(J\) is the relevance judgment set;
- a retrieval system \(r\) returns a ranked list for each query; and
- \(x_{q,r}\in[0,1]\) is a per-query utility such as nDCG@10.

The full score is

\[
\mu_B(r)=\sum_{q\in Q}p_q x_{q,r},
\]

where \(p_q\) is normally uniform within a dataset. A compressed policy at
budget \(b\) selects support \(S_b\subseteq Q\) and nonnegative weights
\(w_{b,q}\) that sum to one:

\[
\widehat{\mu}_{B,b}(r)=\sum_{q\in S_b}w_{b,q}x_{q,r}.
\]

Once ranked runs have been converted to the query-by-system outcome matrix
\(X=[x_{q,r}]\), the compression problem is agnostic to how a system produced
its ranking. The same interface covers:

- BM25 and query expansion;
- learned sparse retrieval;
- dense bi-encoders;
- instruction-conditioned encoders;
- late-interaction models;
- sparse/dense hybrids; and
- cross-encoders when used in a declared candidate-generation and reranking
  pipeline.

A raw cross-encoder over every query-document pair is not required. A
cross-encoder system means a reproducible end-to-end reranking pipeline, such
as BM25 top-100 followed by a fixed cross-encoder.

### Non-goals

This paper is not about:

- model, embedding, index, or document-corpus compression;
- reducing relevance-judgment collection cost;
- claiming that the full benchmark is itself statistically or socially valid;
- finding one static subset that is optimal for every future retriever;
- making recursive self-improvement the source of novelty; or
- replacing full evaluation for final claims without a declared reliability
  target.

The existing 40,159-query benchmark in the
[agentic benchmark release](https://github.com/yzhu319/agentic_research_benchmarks/tree/main/benchmarks/benchmark-compression-optimization)
is useful pilot infrastructure. Its visible-to-sealed gap motivates the study,
but it is custom-built and must not be the paper's main evidence.

## Precise definitions

### Evaluation budget and savings

For a system \(r\), write its cost as

\[
C_r(Q)=C_r^{\mathrm{fixed}}+\sum_{q\in Q}c_r(q).
\]

Index construction and model loading are fixed costs. Query encoding, search,
and reranking are variable costs. Query selection can reduce only the second
term.

The primary budget is the retained fraction of variable evaluation cost:

\[
b_r(S)=
\frac{\sum_{q\in S}c_r(q)}
     {\sum_{q\in Q}c_r(q)}.
\]

When query costs are treated as equal, \(b=|S|/|Q|\). A 5% budget therefore
means 95% fewer query executions, not necessarily 95% less end-to-end
wall-clock time. The paper must report both this normalized budget and measured
wall-clock/GPU savings under explicit index-amortization assumptions.

Report three workload views separately:

1. **Index already available:** query encoding, search, and metric cost only.
2. **Fresh end-to-end system:** document encoding and index construction are
   included, so query compression may save much less.
3. **Reranking pipeline:** candidate generation is fixed and query-document
   reranking cost scales with the selected queries and candidate depth.

Never translate a 5% retained query budget into “95% cheaper” without the
measured workload-specific result.

### The decision to preserve

For systems \(r_i,r_j\), the full-benchmark difference is

\[
\Delta_{ij}=\mu_B(r_i)-\mu_B(r_j),
\]

and the compressed difference is \(\widehat{\Delta}_{ij}\). The primary
decision is the sign of this difference. The primary fidelity metric is
pairwise decision agreement:

\[
A_{\mathrm{pair}}(S;R)=
\frac{1}{\binom{|R|}{2}}
\sum_{i<j}
\mathbb{1}\left[
\operatorname{sign}(\widehat{\Delta}_{ij})
=\operatorname{sign}(\Delta_{ij})
\right].
\]

For a shift experiment, define the evaluated pair set before training. The
primary set contains every comparison with at least one held-out system.
Report held-out–reference and held-out–held-out comparisons separately; the
latter is the cleanest test when a fold contains at least two systems. Replace
the sum and denominator above by this declared pair set.

Near-ties must not be hidden. Results will also be stratified by the magnitude
of \(|\Delta_{ij}|\), and repeated with pre-registered practical tie margins.
Score error, Kendall rank correlation, winner agreement, and top-\(k\)
membership are secondary outcomes.

### Fidelity versus reliability

These terms must remain distinct throughout the paper:

- **Fidelity** is observed agreement with the full benchmark on one fixed
  panel of retrievers.
- **Reliability** is expected out-of-sample fidelity for a declared target
  population of retrievers, estimated using held-out groups and reported with
  uncertainty.

In plain language, fidelity asks, “Did this subset reproduce these systems?”
Reliability asks, “How often should it reproduce the conclusion for another
system drawn from the use case we claimed?”

### Compression–reliability frontier

Let \(\mathcal P\) denote the target retriever population, \(\mathcal D\) the
decision rule, and \(\mathcal A\) a compression procedure that may use only its
development systems. Its frontier is

\[
F_{\mathcal A}(b\mid B,\mathcal P,\mathcal D)
=
\mathbb E\left[
A_{\mathrm{pair}}\left(
\mathcal A(B,R_{\mathrm{dev}},b);R_{\mathrm{test}}
\right)
\right],
\]

where development and test systems are separated according to the declared
shift regime. The expectation is estimated across held-out retriever groups,
datasets, and method seeds. The empirical upper envelope across eligible
methods estimates the currently attainable frontier; it is not claimed to be
the unknowable global optimum.

For target reliability \(\rho\), define the required budget

\[
b^*(\rho)=\inf\{b:F(b)\geq\rho\}.
\]

This turns a plot into an actionable answer: for example, “retain 10% of query
cost to achieve 95% held-out pairwise agreement for these retriever families.”
For deployment, choose the first budget whose **lower confidence bound**, not
just point estimate, reaches the target reliability.

### Price of retriever coverage

For a narrow population \(\mathcal P_1\) and a broader one
\(\mathcal P_2\supset\mathcal P_1\), define

\[
\operatorname{Price}_{\rho}(\mathcal P_1\rightarrow\mathcal P_2)
=b^*_{\mathcal P_2}(\rho)-b^*_{\mathcal P_1}(\rho).
\]

The ratio of the two budgets should also be reported. This quantity measures
how much more evaluation is needed to support broader claims about future
retrievers.

## Research questions and falsifiable hypotheses

### RQ1: What do retrieval frontiers look like?

How does reliability change from 1% to 100% retained query cost across
benchmarks, metrics, and decision margins? Which observable benchmark
properties predict compressibility?

### RQ2: Does retriever shift invalidate fitted microbenchmarks?

Compare random-system, checkpoint, lineage, mechanism, pipeline, and temporal
holdouts. The key hypothesis is:

> A subset's visible-panel fidelity systematically overestimates its
> reliability under lineage and mechanism shift, especially at small budgets.

This is a hypothesis to test, not wording to put in the abstract as an
established fact.

### RQ3: Can an automatic compressor improve the frontier?

Does AutoCompress-IR achieve higher held-out decision agreement, or reach a
fixed reliability target with less cost, than random sampling and prior
selection methods?

### RQ4: What determines transfer?

Does transfer depend more on the number of construction systems, their
behavioral diversity, their named family labels, or query features? Can the
distance of a new retriever from the reference behavioral span predict
compression failure?

### RQ5: Do nominal savings become real savings?

How do query-count reductions translate to wall-clock, accelerator, energy,
and monetary savings for sparse, dense, late-interaction, and reranking
pipelines when fixed indexing costs are included or amortized?

## Proposed contributions

The final paper should make at most these five claims, with numerical language
filled in only after the sealed evaluation:

1. **Scientific object.** Define the compression–reliability frontier and the
   price of retriever coverage for retrieval evaluation.
2. **Empirical diagnosis.** Provide a controlled map of benchmark compression
   under checkpoint, lineage, mechanism, pipeline, and temporal retriever
   shift, and establish the exact novelty claim only after the literature
   audit.
3. **Method.** Introduce AutoCompress-IR, which converts a benchmark, a
   reference run panel, retriever metadata, and a cost target into nested
   weighted query policies with held-out reliability estimates.
4. **Theory.** Separate exact finite-panel fitting from unseen-retriever
   transfer, give a distribution-free sampling guarantee, and bound transfer
   error by behavioral coverage of the reference panel.
5. **Artifact.** Release versioned run files, per-query outcomes, metadata,
   compressed policies, costs, and frontier-generation code so the central
   results can be reproduced without repeating all retrieval inference.

The core contribution is the combination of the first four. A release of
query subsets by itself is not enough.

## AutoCompress-IR

### Inputs

AutoCompress-IR takes:

1. a benchmark: queries, qrels, dataset strata, and target metric;
2. canonical run files or a per-query utility matrix for reference systems;
3. retriever metadata: mechanism, model lineage, checkpoint, training regime,
   pipeline, and release date;
4. one or more retained-cost budgets;
5. a target decision, normally pairwise ordering under nDCG@10;
6. constraints such as nesting, nonnegative weights, minimum dataset coverage,
   and maximum support; and
7. a confidence level or target reliability.

### Outputs

It emits:

- one ordered query policy whose prefixes realize all requested budgets;
- a nonnegative weight vector for each prefix;
- an unweighted-policy variant for easier deployment;
- cross-validated reliability estimates and confidence intervals at each
  budget;
- the smallest estimated budget meeting the requested reliability target; and
- a manifest recording inputs, splits, costs, seeds, and software revisions.

The output is a policy for a stated retriever population, not a universal
benchmark replacement.

### Behavioral coreset formulation

For each reference-system pair \(p=(i,j)\), define a per-query difference

\[
z_{q,p}=x_{q,r_i}-x_{q,r_j}.
\]

Each query is therefore represented by the model comparisons it supports or
opposes. The full mean of each coordinate gives a pairwise system difference.
The compressor seeks a sparse weighted approximation that preserves these
means and, more importantly, their signs.

The training objective should combine:

- a smooth surrogate for pairwise decision error, weighted toward close
  decisions without ignoring easy pairs;
- normalized aggregate-score error;
- a worst-group or distributionally robust term across retriever groups;
- penalties for instability across group-held-out folds; and
- explicit dataset-coverage and nesting constraints.

Exact pairwise agreement remains the evaluation metric. The smoother objective
is only an optimization device.

### First implementation

Build the first credible version before exploring elaborate search:

1. Construct the per-query utility and pair-difference matrices.
2. Keep all checkpoints from one lineage in the same cross-validation fold.
3. Learn continuous nonnegative weights with a conditional-gradient
   (Frank–Wolfe) or projected sparse optimization step.
4. Convert the support to a single priority ordering and deterministically
   round it at all budgets.
5. Improve rounded prefixes with budget-aware add/swap moves against the
   worst held-out-development fold.
6. Select hyperparameters only by grouped nested cross-validation.
7. Calibrate a lower confidence bound for reliability at every budget.

For small datasets, solve a mixed-integer version to obtain an oracle
optimization-gap diagnostic. It may inspect the development panel only.

An outer-loop/RSI search can be evaluated later as another optimizer over the
same valid policy space and evaluation budget. It earns space in the paper
only if it improves the held-out frontier; it is not required for the main
story.

### Optional progressive evaluation

The earlier phrase “confidence-based escalation” means something simple:

> Start with a small prefix. If two systems are too close to call under the
> estimated error interval, evaluate the next prefix. Close calls receive more
> queries; clear decisions stop early.

The nested policy makes this possible without discarding prior work. Call it
**progressive evaluation** in the paper. Treat it as a secondary application,
not part of the definition of the frontier.

## Theory agenda

The theory must explain the central empirical problem rather than decorate the
paper.

### T1. Exact fitting of a finite visible panel

For \(K\) known retrievers, let

\[
v_q=(x_{q,r_1},\ldots,x_{q,r_K})\in[0,1]^K.
\]

The full score vector is the mean of the \(v_q\)'s and therefore lies in their
convex hull. By Carathéodory's theorem, there exists a nonnegative weighted
subset of at most \(K+1\) queries that exactly reconstructs all \(K\) aggregate
scores.

This result is important because it explains why spectacular compression on a
visible panel can be meaningless. It guarantees fit, not transfer. State all
qualifications:

- it permits nonuniform weights;
- it says nothing about an unseen retriever;
- multiple separately constrained datasets or metrics increase the effective
  dimension; and
- finding a useful nested or uniform subset is a different problem.

### T2. No-free-lunch result for unrestricted unseen retrievers

For any strict query subset, construct two unseen retrievers that agree on the
selected queries but differ on omitted queries enough to reverse their
full-benchmark order. Therefore no deterministic strict subset can guarantee
decision preservation for an unrestricted future retriever class.

This establishes that every transfer claim requires an explicit population or
structural assumption.

### T3. Distribution-free random-sampling guarantee

For bounded per-query utilities and \(K\) fixed retrievers, uniform sampling
with

\[
n=O\left(\frac{\log(K/\delta)}{\epsilon^2}\right)
\]

queries estimates all \(K\) aggregate scores to error \(\epsilon\) with
probability at least \(1-\delta\). Pairwise signs are then preserved for full
benchmark margins larger than \(2\epsilon\).

Provide constants, finite-population correction, stratified extension, and a
matching lower-bound discussion. This is the honest baseline any learned
compressor must beat.

### T4. Behavioral-coverage transfer bound

Let \(u_r\in\mathbb R^{|Q|}\) be the per-query utility vector of an unseen
retriever and \(U\) the matrix of reference utility vectors. For full weights
\(p\) and compressed weights \(w\), decompose

\[
u_r=U\alpha+e.
\]

Then

\[
|(w-p)^\top u_r|
\leq
\|\alpha\|\,\|(w-p)^\top U\|
+
\|w-p\|_*\,\|e\|.
\]

The first term is visible-panel approximation error. The second is the unseen
retriever's distance from the reference behavioral span. This formalizes the
claim that more behaviorally diverse references can improve transfer, while
mere checkpoint count may not. State the bound with an explicit pair of
compatible dual norms in the proof; the display above is shorthand for that
choice.

Turn this into a computable diagnostic using held-out residual distance or a
regularized projection. Test whether that diagnostic predicts actual errors.

### T5. Optional optimization hardness

If a clean reduction is available, show that minimum-support uniform or nested
decision-preserving selection is NP-hard, motivating approximation algorithms.
This is optional. Do not spend the theory budget on it before T1–T4 are
complete and checked.

### Theory definition of done

- Every theorem has assumptions matching an experiment.
- T1–T4 have independent proof checks and synthetic sanity tests.
- The transfer bound yields a measured quantity or testable prediction.
- No asymptotic result is presented as a practical saving without constants.

## Empirical design

### Public benchmark suite

The main suite uses all English BEIR entries with at most 600,000 documents:

| BEIR entry | Documents | Test queries | Role |
|---|---:|---:|---|
| TREC-COVID | 171,332 | 50 | very small query set; biomedical |
| NFCorpus | 3,633 | 323 | small corpus; biomedical |
| FiQA-2018 | 57,638 | 648 | finance |
| ArguAna | 8,674 | 1,406 | argument retrieval |
| Touche-2020 | 382,545 | 49 | very small query set; argument retrieval |
| CQADupStack | 457,199 | 13,145 | twelve heterogeneous CQA tasks |
| Quora | 522,931 | 10,000 | duplicate-question retrieval |
| SciDocs | 25,657 | 1,000 | scientific document links |
| SciFact | 5,183 | 300 | scientific claim evidence |
| **Total** | **1,634,792** | **26,921** | before deduplication across entries |

Verify these counts against the pinned BEIR/ir_datasets revision and preserve
the twelve CQADupStack task boundaries. Do not silently change the suite after
seeing results.

Add Natural Questions as a large-corpus stress test. It should not control
method selection. Its purpose is to show how query reduction interacts with a
roughly 2.7-million-document index and therefore with fixed versus variable
cost.

The primary metric is nDCG@10. Recall@100 and MRR@10 are secondary where
defined. Report each dataset separately and an equal-dataset macro average;
never let Quora or CQADupStack dominate merely because they have more queries.

### Retriever population

“Retriever family” is not a single objective label. Record at least five
overlapping axes:

1. retrieval mechanism: lexical, learned sparse, dense, late interaction,
   hybrid, or reranking;
2. checkpoint lineage: models sharing architecture and training ancestry;
3. supervision/training data: unsupervised, MS MARCO supervised, contrastive,
   distillation, or instruction tuned;
4. pipeline composition: first-stage model, fusion, candidate depth, and
   reranker; and
5. release time.

All checkpoints from one lineage must stay in the same split. Family metadata
must be frozen before outcome analysis.

The target is 20–24 core systems plus 4–6 sealed systems. The following is a
candidate registry, not permission to substitute whichever systems give the
best story:

| Mechanism | Candidate systems |
|---|---|
| Lexical | BM25; BM25+RM3 |
| Learned sparse | uniCOIL; SPLADE++; one independently trained SPLADE lineage |
| Earlier dense | DPR; ANCE; TAS-B; Contriever |
| Modern dense | E5 base/large; BGE base/large; GTE base/large |
| Instruction-conditioned | INSTRUCTOR; Nomic Embed; Snowflake Arctic Embed |
| Late interaction | ColBERTv2; a second independently trained late-interaction checkpoint if reproducible |
| Hybrid | BM25+E5 reciprocal-rank fusion; BM25+SPLADE fusion; sparse+dense learned or fixed fusion |
| Reranking pipelines | BM25 top-100 + MiniLM cross-encoder; BM25 top-100 + BGE reranker; dense top-100 + fixed reranker |

Before full runs:

- smoke-test every candidate on two datasets;
- verify license, public availability, deterministic inference, memory, and
  query/document formatting;
- select exact model and code revisions;
- record index and run checksums in systems.lock.yaml; and
- freeze the panel before evaluating compression methods.

Do not use proprietary or API-only retrievers in the primary claims. Sealed
means withheld from compressor development, not unavailable to readers. All
sealed run files and manifests are released after final evaluation.

Generating this panel is the expensive part of the project, but it happens
once. Cap the panel rather than chasing every leaderboard model. If compute is
tight, preserve mechanism and lineage diversity before preserving raw model
count; a minimum viable full study is 16 core plus 4 sealed systems.

### Shift regimes

Use increasingly difficult split regimes:

| Regime | What is held out | Purpose |
|---|---|---|
| Random-system | arbitrary systems | optimistic negative control |
| Checkpoint | size, seed, or nearby checkpoint from a seen lineage | local interpolation |
| Lineage | all checkpoints from one model lineage | unseen training ancestry |
| Mechanism | an entire sparse, dense, late-interaction, hybrid, or reranking group | structural extrapolation |
| Pipeline | a full fusion/reranking composition | system-level extrapolation |
| Temporal | later model-release cohorts | realistic future-system test |

Lineage and mechanism shift are the primary tests. Run a retrospective temporal
split by training on older public models and testing on newer public models.
If suitable models appear after the protocol freeze, add them as a prospective
sealed cohort. Temporal shift is confirmatory because it has fewer samples and
can be confounded by calendar time.

Use grouped cross-validation within the core panel. Keep 4–6 additional
systems sealed until the method, hyperparameters, plots, and claim thresholds
are committed. Unlock the sealed outcome matrix once. Archive the unlock
record and then release the matrix publicly.

### Budgets

Evaluate retained variable-cost budgets:

\[
\{1,2,5,10,20,40,60,80,100\}\%.
\]

For equal-cost analysis, use the exact integer count implied by each dataset
and report rounding. Very small test sets will make the 1% and 2% points
intentionally unstable. For measured-cost analysis, select under profiled
per-query costs.

Policies should be nested so work at a smaller budget is reused at a larger
one. Also run independently optimized non-nested policies as an upper-bound
ablation.

### Required baselines

Every baseline receives the same development systems and no sealed outcomes:

1. uniform random sampling, with at least 100 draws per budget;
2. stratified random sampling by dataset and available query labels;
3. query-feature k-medoids;
4. prior IR topic-selection methods, including *A Few Good Topics*,
   uncertainty-aware selection, and intelligent topic selection where their
   assumptions apply;
5. Anchor Points;
6. tinyBenchmarks;
7. visible aggregate-mean matching;
8. supervised greedy pairwise-margin preservation;
9. recent ranking-preserving dataset-selection and coreset methods, when code
   and licenses permit;
10. the proposed AutoCompress-IR method; and
11. a test-panel oracle used only to show headroom, clearly labeled invalid for
    deployment.

Active sampling methods designed to reduce relevance judging should be
included only in the related-work comparison or adapted transparently; do not
pretend their cost model is identical.

### Primary and secondary outcomes

Primary:

- held-out pairwise decision agreement under nDCG@10;
- required budget \(b^*(0.95)\) for 95% reliability; and
- worst-dataset and worst-shift-group reliability.

Secondary:

- aggregate-score mean and maximum absolute error;
- Kendall's \(\tau\) and Spearman rank correlation;
- winner and top-\(k\) membership agreement;
- calibration of predicted reliability;
- reliability by full-benchmark pair margin;
- support overlap and policy stability across seeds;
- wall-clock, GPU-hours, energy if available, and monetary cost; and
- storage and one-time indexing cost.

Report the entire frontier. A single favorable budget is not sufficient.

### Experiment matrix

| ID | Experiment | Decision enabled |
|---|---|---|
| E0 | Build and audit all full run files | establishes the full-evaluation target |
| E1 | Random and fitted frontiers on fixed panels | measures basic compressibility and visible overfit |
| E2 | Checkpoint, lineage, and mechanism shift matrix | tests whether family shift is real and how serious it is |
| E3 | AutoCompress-IR versus all baselines | tests the algorithmic contribution |
| E4 | Vary construction-panel size and diversity | estimates the price and value of retriever coverage |
| E5 | Behavioral-distance prediction | tests the transfer-bound mechanism |
| E6 | Sealed and temporal systems | supplies the final non-adaptive test |
| E7 | Measured end-to-end cost | translates nominal compression into practical savings |
| E8 | Progressive evaluation | tests whether close-call expansion saves more cost |

### Required ablations

- uniform versus learned weights;
- nested versus independently optimized policies;
- average-risk versus worst-group optimization;
- random folds versus lineage-grouped folds;
- outcome vectors versus query features versus both;
- with and without retriever-family metadata;
- number of reference systems at fixed mechanism diversity;
- mechanism diversity at fixed reference-system count;
- pairwise-decision loss versus score-reconstruction loss;
- practical tie-margin choices;
- per-dataset compression versus global suite-level budget allocation; and
- greedy/local search versus optional outer-loop search at matched candidate
  evaluation count.

### Statistical protocol

- Pre-register the primary metric, shifts, budgets, tie-margin sensitivity,
  and go/no-go thresholds before sealed evaluation.
- Use identical grouped folds and random seeds for paired method comparisons.
- Bootstrap at the retriever-lineage and dataset level. Retriever pairs sharing
  a system are not independent observations.
- Report 95% confidence intervals and raw per-dataset values.
- Use at least 100 random-subset replicates and at least 10 method seeds where
  the algorithm is stochastic.
- Plot raw frontier estimates. An isotonic curve may summarize the expected
  monotone trend, but it must not replace raw values.
- Report failures and variance at tiny budgets, not only means.
- Choose hyperparameters through nested grouped validation; never through
  sealed systems.
- Treat full-benchmark scores as the target being reproduced, not as noiseless
  ground truth. Include paired-bootstrap uncertainty of the full benchmark as
  context.

## Reproducibility and cost contract

Others should not need to rerun 25 expensive retrievers to verify the
compression claims.

### Lightweight reproduction

Release:

- canonical TREC-format ranked-run files;
- qrels and exact query IDs;
- per-query metric matrices;
- retriever metadata and split assignments;
- query costs and profiling logs;
- every selected policy and weight vector;
- all random seeds and environment locks; and
- one command that regenerates every frontier, table, and statistical test
  from static artifacts.

This path should run on CPU in hours or less.

### End-to-end audit

Define an eight-system audit panel spanning BM25, learned sparse, classic
dense, modern dense, instruction-conditioned, late interaction, hybrid, and
reranking mechanisms. Provide scripts to recreate its indexes and runs on all
main datasets. The remaining systems are reproducible from pinned open models
but need not be rerun for an ordinary paper reproduction.

Publish container hashes, package locks, model revisions, prompt prefixes,
tokenization, pooling, normalization, candidate depths, fusion constants,
hardware, and run checksums. Avoid mutable “latest” model references.

### Suggested artifact layout

~~~text
papers/benchmark-compression-for-retriever-evaluation/
├── README.md
├── protocol.md
├── systems.lock.yaml
├── datasets.lock.yaml
├── configs/
├── runs/
├── outcomes/
├── policies/
├── results/
├── scripts/
└── paper/
~~~

Large run artifacts may live in a versioned external release, with checksums
and download scripts kept here.

## Novelty boundary and ICLR bar

Query/topic selection for cheaper IR evaluation is established work. General
benchmark compression, anchor selection, model-performance prediction, and
ranking-preserving coresets are also active areas. The paper must not claim to
invent benchmark subsets or ranking preservation.

The intended novelty is the conjunction of:

1. treating **out-of-retriever reliability** as the central quantity;
2. mapping full cost–reliability frontiers rather than reporting one subset;
3. separating checkpoint, lineage, mechanism, pipeline, and temporal shifts;
4. automatically constructing nested policies for a declared target
   population;
5. proving why finite-panel fit can be arbitrarily misleading and when
   behavioral coverage controls transfer; and
6. releasing a broad, static, reproducible run panel.

For an ICLR submission, the paper needs all of the following:

- a material and reproducible transfer phenomenon;
- an algorithmic gain at fixed reliability or fixed cost;
- theory that predicts or explains the empirical result;
- breadth across datasets and retrieval mechanisms; and
- a reusable artifact whose claims can be reproduced cheaply.

Without the shift study and theory, this is likely an incremental IR
topic-selection paper. Without an algorithmic gain, it may still become a
strong reliability/limits paper if the failure phenomenon and explanatory
theory are unusually clear. Without either, stop or redirect to a smaller IR
venue.

The connection to outer-loop research is practical, not rhetorical: reliable
cheap evaluation allows a model-development loop to compare more candidates
without silently changing its objective. No agent-specific assumption is
needed for the contribution.

## Proposed abstract

> Retrieval benchmarks are repeatedly used to choose among rapidly changing
> sparse, dense, late-interaction, hybrid, and reranking systems, yet running
> every query for every candidate can dominate development cost. Existing
> benchmark-compression methods typically return one small subset and assess it
> on systems available during construction. We ask a different question: how
> small can a retrieval benchmark become while preserving the decisions of the
> full benchmark for retrievers not used to compress it? We formalize this
> tradeoff as the **compression–reliability frontier**, where reliability is
> held-out agreement on pairwise system comparisons under declared checkpoint,
> lineage, mechanism, pipeline, and temporal shifts. We introduce
> **AutoCompress-IR**, a behavioral-coreset method that produces nested,
> cost-aware query policies and an estimated budget for a target reliability.
> We also separate finite-panel fit from transfer: a weighted subset of at most
> \(K+1\) queries can exactly match \(K\) known retrievers, while no strict
> subset can guarantee an unrestricted unseen retriever; a coverage bound
> relates transfer error to distance from the reference behavioral span.
> Across [DATASETS] and [SYSTEMS], we find [SHIFT RESULT]. At [RELIABILITY]
> reliability, AutoCompress-IR retains [BUDGET] of variable evaluation cost,
> improving over [STRONGEST BASELINE] by [RESULT], with policies, run files,
> and full frontier-generation code released for reproduction.

All bracketed fields are mandatory result placeholders. Do not replace them
with qualitative claims until the sealed run is complete.

## Paper outline

1. **Introduction:** evaluation cost, the danger of a universal tiny
   benchmark, and the frontier question.
2. **Problem formulation:** ranked-run abstraction, cost, decisions, fidelity,
   reliability, frontier, and coverage price.
3. **Why visible fit is insufficient:** exact-fit theorem, no-free-lunch
   result, sampling guarantee, and transfer bound.
4. **AutoCompress-IR:** behavioral representation, robust grouped objective,
   nested selection, and reliability calibration.
5. **Experimental protocol:** BEIR suite, retriever registry, shifts,
   baselines, leakage controls, and costs.
6. **Results:** frontier maps, family-shift audit, method comparison, coverage,
   and behavioral-distance analysis.
7. **Practical use:** target-reliability policy selection and optional
   progressive evaluation.
8. **Related work and limitations:** IR topic selection, benchmark
   compression, model shift, qrel limitations, and environmental cost.
9. **Conclusion:** compression is a conditional reliability decision, not a
   fixed subset.

## Execution plan

### Phase 0 — Freeze the question and protocol

- [ ] Copy the definitions in this document into protocol.md.
- [ ] Confirm the official ICLR 2027 timeline and work backward from it.
- [ ] Complete a claim-by-claim related-work table.
- [ ] Verify all dataset counts and licenses.
- [ ] Define the exact pairwise decision rule and tie-margin sensitivity.
- [ ] Pre-register budgets, primary shifts, confidence intervals, and pilot
      gates.
- [ ] Create datasets.lock.yaml and an initial systems.lock.yaml.
- [ ] Freeze code/data boundaries so sealed outcome files cannot be loaded by
      development jobs.

**Deliverable:** a versioned protocol whose primary analysis cannot be changed
after seeing sealed results.

### Phase 1 — Build the static evaluation panel

- [ ] Implement one run adapter and per-query metric schema.
- [ ] Smoke-test every candidate system on TREC-COVID and SciFact.
- [ ] Select 20–24 core and 4–6 sealed open systems.
- [ ] Generate or acquire full ranked runs for the nine main BEIR entries.
- [ ] Preserve CQADupStack constituent-task boundaries.
- [ ] Profile fixed and per-query costs for every mechanism.
- [ ] Validate checksums, score direction, query counts, and metric parity.
- [ ] Implement the eight-system end-to-end audit path.

**Deliverable:** immutable run and per-query outcome matrices plus cost and
lineage metadata.

### Phase 2 — Run the decisive pilot

- [ ] Use TREC-COVID, NFCorpus, FiQA, Quora, and SciFact.
- [ ] Use at least 12 systems spanning six mechanisms and multiple lineages.
- [ ] Run random, stratified random, visible mean matching, Anchor Points,
      tinyBenchmarks, greedy pairwise preservation, and AutoCompress-IR.
- [ ] Measure 1%, 2%, 5%, 10%, and 20% budgets.
- [ ] Compare random-system, checkpoint, lineage, and mechanism holdouts.
- [ ] Plot visible fidelity beside held-out reliability.
- [ ] Run the go/no-go review below before scaling.

**Deliverable:** one compact evidence packet that decides whether the ICLR
story exists.

### Phase 3 — Complete AutoCompress-IR

- [ ] Implement behavioral difference vectors and grouped folds.
- [ ] Implement continuous sparse weighting and deterministic rounding.
- [ ] Implement budget-aware local swaps and nesting.
- [ ] Add worst-group optimization and reliability calibration.
- [ ] Add per-dataset coverage constraints and cost-aware selection.
- [ ] Build a small-instance mixed-integer oracle.
- [ ] Run every required ablation.
- [ ] Freeze method and hyperparameters before sealed evaluation.

**Deliverable:** a documented command that accepts benchmark artifacts and
emits policies plus a frontier report.

### Phase 4 — Prove and test the theory

- [ ] Write formal statements and proofs for T1–T4.
- [ ] Check edge cases: weights, multiple metrics, strata, ties, and finite
      populations.
- [ ] Build synthetic examples that attain exact fit but fail under shift.
- [ ] Estimate behavioral-span distance for every held-out retriever.
- [ ] Test whether distance predicts score and decision errors.
- [ ] Seek an independent proof review.
- [ ] Attempt T5 only after the core theory is complete.

**Deliverable:** theory that explains a measured phenomenon and survives an
independent check.

### Phase 5 — Full evaluation and one-time seal

- [ ] Run all methods on all budgets and datasets using frozen grouped folds.
- [ ] Generate frontiers and coverage-price curves.
- [ ] Complete measured-cost and NQ stress-test studies.
- [ ] Commit the final code, method choice, and analysis notebook.
- [ ] Unlock the 4–6 sealed systems exactly once.
- [ ] Run the pre-registered sealed analysis without method changes.
- [ ] Release sealed run files and record the unlock.
- [ ] If a bug requires rerunning, document it and label the sealed panel
      compromised rather than silently resetting it.

**Deliverable:** final numbers with a complete provenance trail.

### Phase 6 — Write and release

- [ ] Fill abstract placeholders only from the frozen result tables.
- [ ] Lead the paper with the frontier and the shift result, not the optimizer.
- [ ] Put every per-dataset and per-shift result in the appendix/artifact.
- [ ] Include limitations about family labels, qrels, temporal sample size,
      corpus indexing, and full-benchmark uncertainty.
- [ ] Package the CPU-only reproduction path.
- [ ] Reproduce every figure and table from a clean environment.
- [ ] Run an internal skeptical review against the ICLR bar.

**Deliverable:** submission, public artifacts, and an auditable result ledger.

## Pilot go/no-go criteria

Set these thresholds before running the pilot to avoid rationalizing a weak
signal.

### Gate A: the reliability problem exists

Pass if, at 5% or 10% retained cost, either:

- lineage/mechanism holdout reduces pairwise agreement by at least 3 absolute
  percentage points relative to checkpoint holdout, with a clustered 95%
  interval excluding zero; or
- broadening the target population at least doubles the estimated budget
  needed for 95% reliability.

The effect should appear across multiple datasets and more than one fitted
compression method, not depend on a single pathological system pair.

### Gate B: the proposed method adds value

Pass if AutoCompress-IR, against the strongest legitimate baseline, either:

- improves held-out pairwise agreement by at least 2 absolute points at a
  budget of 10% or less; or
- reduces the budget required for 95% reliability by at least a factor of 1.5,

without losing more than one point on worst-dataset reliability. Confirm the
same direction on most main datasets in the full study.

### Gate C: the savings are operationally meaningful

Pass if a 10% query policy yields at least a fivefold measured speedup for an
index-reuse or reranking workload and the paper can name a realistic repeated
evaluation setting where fixed costs do not erase the benefit. Report the
fresh-index case even if it shows little end-to-end saving.

### Gate D: the result is practically reproducible

Pass if the full run panel can be frozen, the central analyses regenerate from
static artifacts on CPU, and the audit panel can be rebuilt from pinned public
models with matching run checksums or documented numerical tolerance.

### Decision

- **A + B + C + D pass:** proceed as the intended ICLR 2027 paper.
- **A passes, B fails:** consider a limits/reliability paper only if the theory
  strongly predicts the failures; otherwise target an IR venue.
- **B passes, A fails:** this is a benchmark-compression method paper, but the
  family-shift motivation should be removed rather than exaggerated.
- **C fails:** retain the statistical contribution only if it is strong, and
  remove broad compute-saving claims.
- **A or D fails and no strong replacement result appears:** stop or narrow
  the project.

The numerical gates are planning thresholds, not claims of statistical or
practical significance for every use case. Freeze or revise them, with written
reasoning, before looking at pilot outcomes.

## Handoff rules for the executing agent

1. Start with Phase 0 and the pilot. Do not launch the full matrix before the
   go/no-go review.
2. Maintain a result ledger with one row per sentence-level claim, its script,
   artifact checksum, table/figure, and status.
3. Never tune on sealed outcomes, including by manually viewing their summary
   statistics.
4. Keep failed methods and negative datasets in the artifact.
5. Distinguish exploratory plots from frozen confirmatory analyses.
6. Record every panel or protocol change in Git before rerunning.
7. Prefer public run artifacts and deterministic pipelines over a larger but
   irreproducible system count.
8. Do not write “retriever-family shift breaks compression” unless Gate A
   passes. Until then, write “we test whether it does.”
9. Do not claim a global optimum. Report an empirical method frontier and an
   oracle only where its information access is explicit.
10. Keep the main message crisp: **the smallest safe benchmark depends on the
    reliability target and the retrievers it must generalize to.**

## Primary related work

- [A Few Good Topics: Experiments in Topic Set Reduction for Retrieval Evaluation](https://doi.org/10.1145/1629096.1629099)
- [Uncertainty-aware query selection for evaluation](https://doi.org/10.1145/2348283.2348403)
- [Intelligent topic selection for information retrieval evaluation](https://doi.org/10.1016/j.ipm.2017.09.002)
- [Active Sampling for Large-scale Information Retrieval Evaluation](https://arxiv.org/abs/1709.01709)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- [Anchor Points: Benchmarking Models with Much Fewer Examples](https://arxiv.org/abs/2309.08638)
- [tinyBenchmarks: Evaluating LLMs with Fewer Examples](https://arxiv.org/abs/2402.14992)
- [Microbenchmark reliability (ICLR 2026)](https://arxiv.org/abs/2510.08730)
- [EssenceBench](https://arxiv.org/abs/2510.10457)
- [Ranking-preserving dataset selection (KDD 2026)](https://arxiv.org/abs/2606.27997)
- [Coresets Before Score Sets](https://arxiv.org/abs/2607.09739)

The first literature task is to build a claim matrix for these papers: setting,
unit of compression, cost model, target statistic, whether systems are held
out by lineage/mechanism/time, theory, and released artifacts. The novelty
section should be rewritten from that matrix rather than from memory.
