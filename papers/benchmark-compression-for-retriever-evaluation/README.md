# The Compression–Reliability Frontier: How Compressible Is a Retrieval Benchmark? (research proposal)

Status: research proposal under reassessment, originally developed as a handoff
and execution plan for a possible ICLR 2027 submission.
No paper experiments described below have been run unless explicitly marked as
pilot evidence.

The design and recommendations below preserve the original proposal. The
subsequent [frontier assessment](frontier-assessment.md) finds insufficient
novelty in the current framing and unresolved reliability claims; it does not
recommend executing the full plan yet. Read the assessment alongside the
[alternative research directions and ratings](research-directions.md) before
choosing a pilot or committing to a paper.

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
want a small set of representative queries. Existing methods generally start
from a chosen subset size and return one compressed benchmark. That leaves the
practical question unanswered: was that size small enough to save meaningful
work, yet large enough to preserve the decisions of the full benchmark?

We formulate this missing tradeoff as the **compression–reliability
frontier**. At every retained-cost budget, the frontier records the best
held-out decision reliability attained by an eligible compression method.
**AutoCompress-IR** estimates this curve, produces nested weighted query
policies, and returns the smallest tested budget whose reliability confidence
bound reaches a user-chosen target. The empirical study asks how these
frontiers differ across retrieval benchmarks and how much AutoCompress-IR
improves them over random sampling and prior compressors.

The intended outcome is both scientific and practical:

- a frontier that says how much evaluation cost is actually safe to remove;
- an algorithm that creates a benchmark policy for a declared reliability
  target and retriever population;
- upper and lower limits on how small a reliable evaluation can be;
- canonical run files that let others reproduce the paper without rerunning
  every retriever.

## Why this matters

A microbenchmark is often used to choose a winner among retrieval systems. A
wrong ordering is more consequential than a small average score error, but a
compression ratio by itself gives no assurance about that decision. Five
percent may be ample for one benchmark and unreliable for another because
query count, redundancy, system disagreement, and full-benchmark decision
margins differ.

The missing scientific and practical object is therefore a calibrated answer
to two linked questions:

1. At a given cost, how reliably can a compressed benchmark reproduce the
   full benchmark's system comparisons?
2. For a required reliability, what is the smallest supported cost?

Answering these questions across datasets reveals whether benchmark
compressibility has stable structure or is merely an arbitrary subset-design
choice. It also gives practitioners an explicit rule for choosing a budget
instead of copying a universal ratio. Structured holdouts by lineage or
retrieval mechanism are Priority-1 stress tests; the main paper neither
assumes nor requires them to reduce reliability.

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

For held-out-system evaluation, define the evaluated pair set before training.
The primary set contains every comparison with at least one held-out system.
Report held-out–reference and held-out–held-out comparisons separately; the
latter is available when a fold contains at least two systems. Replace the sum
and denominator above by this declared pair set.

Near-ties must not be hidden. Results will also be stratified by the magnitude
of \(|\Delta_{ij}|\), and repeated with pre-registered practical tie margins.
Score error, Kendall rank correlation, winner agreement, and top-\(k\)
membership are secondary outcomes.

### Fidelity versus reliability

These terms must remain distinct throughout the paper:

- **Fidelity** is observed agreement with the full benchmark on one fixed
  panel of retrievers.
- **Reliability** is expected out-of-sample fidelity for a declared target
  population of retrievers, estimated using held-out systems and reported
  with uncertainty.

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
resampling protocol. Priority 0 uses repeated, balanced random-system folds
plus a frozen final holdout. Structured group holdouts are optional stress
tests. For fixed \(B\), the expectation is estimated across held-out-system
folds and method seeds. Dataset-specific curves are reported separately; the
headline macro curve gives each dataset equal weight.

For a pre-declared eligible method set \(\mathfrak A\), define the attained
frontier

\[
F(b)=\max_{\mathcal A\in\mathfrak A}F_{\mathcal A}(b).
\]

This empirical upper envelope is the best currently attained result, not a
claim about the unknowable global optimum.

For target reliability \(\rho\), define a method's required budget

\[
b^*_{\mathcal A}(\rho)=\inf\{b:F_{\mathcal A}(b)\geq\rho\}.
\]

Dropping the method subscript denotes the budget on the attained frontier.

This turns a plot into an actionable answer: for example, “retain 10% of query
cost to achieve 95% held-out pairwise agreement for this benchmark and
declared system population.” For deployment, choose the first tested budget
whose **simultaneous lower confidence bound**, not just point estimate, reaches
the target reliability. Denote this conservative grid estimate by
\(\widehat b^*_{\mathcal A,\mathrm{LCB}}(\rho)\).

Comparing \(b^*(\rho)\) across benchmarks is a Priority-0 analysis. Comparing
it across narrower and broader retriever populations is a Priority-1 stress
test, not part of the headline claim.

## Headline figure and minimum paper

The paper should be designed around one image:

> **Retrieval benchmarks have different compression–reliability frontiers;
> AutoCompress-IR reaches a chosen reliability with less evaluation.**

If a reviewer remembers only one figure, it should be **Figure 1: The
Compression–Reliability Frontier**. Its two panels focus on method comparison
and benchmark heterogeneity.

### Figure 1A — Reliability as a function of retained cost

- **Horizontal axis:** retained variable evaluation cost at 1%, 2%, 5%, 10%,
  20%, 40%, and 100%. Add a secondary label for the fraction of queries
  removed.
- **Vertical axis:** held-out pairwise decision agreement under nDCG@10.
- **Protocol:** pre-declared balanced random-system cross-validation over the
  core retriever panel, macro-averaged equally across the six datasets.
- **Curves:** uniform or stratified random sampling, the strongest eligible
  prior compressor, and AutoCompress-IR.
- **Frontier:** emphasize the empirical upper envelope of the eligible method
  curves, while preserving the individual curves.
- **Reference line:** 95% reliability, with each method's required budget
  \(\widehat b^*_{\mathcal A,\mathrm{LCB}}(0.95)\) marked.

The intended visual impression is an up-and-left improvement: at the same
cost AutoCompress-IR is more reliable, or at the same reliability it evaluates
less of the benchmark. All methods reach 100% fidelity at 100% cost by
definition.

### Figure 1B — There is no universal safe compression ratio

Use one row for each core dataset: TREC-COVID, NFCorpus, FiQA, ArguAna, Quora,
and SciFact. Plot the required retained budget for 95% reliability for the
same three methods, with uncertainty. If a method does not cross the target
before full evaluation, show that explicitly rather than clipping or dropping
the row.

This panel answers “How compressible is this benchmark?” directly. It should
make both benchmark heterogeneity and any AutoCompress-IR budget reduction
visible. A compact set of six per-dataset frontier plots can appear as Figure
2 or in the appendix; do not crowd them into Figure 1.

### Plotting rules

- Dots are the measured budgets; connecting lines are visual guides, not
  evidence of a continuously observed curve.
- A method-specific line is a **reliability curve**. Reserve **frontier** for
  the upper envelope across eligible policies at a budget.
- Show raw estimates and clustered 95% confidence bands. Do not show only an
  isotonic-smoothed line.
- Use an equal-dataset macro average in Panel A; never weight datasets by query
  count.
- Keep method colors and the 95% target consistent across both panels.
- Do not add every baseline, metric, or stress test to Figure 1. If the
  legend needs explaining, the plot is too busy.
- Freeze the held-out-system protocol and strongest prior method before the
  final evaluation.

Figure 1 succeeds only if a reader can say, without reading the caption:

> “The safe compression level differs by benchmark, and the proposed method
> reaches the same reliability with less evaluation.”

If the data do not support the method half, the paper needs a substantially
stronger limits result or should be redirected; do not force the figure.

### Priority 0: minimum credible ICLR paper

Protect these experiments first:

- **Datasets:** TREC-COVID, NFCorpus, FiQA, ArguAna, Quora, and SciFact. This
  six-dataset core spans 50 to 10,000 queries and several domains while keeping
  the retrieval sweep manageable.
- **Systems:** roughly 16 development systems plus 4 frozen held-out systems,
  covering
  lexical, learned sparse, dense, instruction-conditioned, late-interaction,
  hybrid, and reranking pipelines. These are existing public systems chosen
  for panel coverage and reproducibility, not because any is “new.”
- **Reliability protocol:** repeated balanced random-system folds for method
  development and uncertainty estimation, followed by one frozen held-out
  evaluation after all choices are fixed.
- **Budgets:** 1%, 2%, 5%, 10%, 20%, 40%, and 100%.
- **Methods:** random, the strongest prior method, a strong supervised greedy
  baseline, and AutoCompress-IR. Other inexpensive baselines still belong in
  the appendix.
- **Outcomes:** pairwise decision agreement under nDCG@10, each dataset's
  \(b^*(0.95)\), and the equal-dataset macro frontier. Score error and rank
  correlation are diagnostics.
- **Theory:** finite-panel exact compression, a finite-population
  random-sampling guarantee, margin-dependent sample-complexity limits, and
  valid uncertainty for selecting a budget from the estimated frontier.
- **Practical evidence:** one index-reuse workload, one reranking workload, and
  the static run-file reproduction path.

This scope must produce Figure 1, a compact table of per-dataset required
budgets, and the six per-dataset reliability curves.

### Priority 1: strengthen if time and compute allow

- add Touche-2020, CQADupStack, and SciDocs;
- expand to 20–24 development and 4–6 frozen held-out systems;
- run lineage, mechanism, and pipeline holdout stress tests;
- test construction-panel size and behavioral diversity;
- test the behavioral-span diagnostic and associated transfer bound;
- test the unrestricted-panel no-free-lunch result only as supporting theory;
- study the price of broadening the declared retriever population;
- run independently optimized non-nested policies;
- report secondary retrieval metrics; and
- broaden the end-to-end cost audit.

### Priority 2: cut first

- Natural Questions as a large-corpus stress test;
- temporal holdouts;
- progressive evaluation;
- outer-loop/RSI optimization;
- an NP-hardness result;
- exhaustive energy accounting; and
- large grids of query representations, optimizers, and tie thresholds.

These ideas may be useful follow-up work. None should delay the headline
frontier, benchmark-heterogeneity result, AutoCompress comparison, or frozen
held-out result.

## Research questions and falsifiable hypotheses

### RQ1: What do retrieval frontiers look like?

How does reliability change from 1% to 100% retained query cost across
benchmarks and decision margins? How much does the budget required for 95%
reliability vary across the six core datasets?

### RQ2: Can an automatic compressor improve the frontier?

Does AutoCompress-IR achieve higher held-out decision agreement, or reach a
fixed reliability target with less cost, than random sampling and the
strongest prior selection method?

### RQ3: What controls how small a benchmark can be?

How do the number of queries, per-query system disagreement, redundancy,
effective behavioral dimension, and full-benchmark pairwise margins relate to
the observed frontier? Do the theoretical upper and lower limits explain when
extreme compression is or is not possible?

### RQ4: Do nominal savings become real savings?

How do query-count reductions translate to wall-clock, accelerator, energy,
and monetary savings for sparse, dense, late-interaction, and reranking
pipelines when fixed indexing costs are included or amortized?

### Secondary RQ (Priority 1): How sensitive is the frontier to the system panel?

Repeat selected experiments with lineage, mechanism, and pipeline group
holdouts. This tests whether panel composition changes the estimated frontier;
it does not assume that any group holdout must be harder than a random-system
holdout.

## Proposed contributions

The final paper should make at most these five claims, with numerical language
filled in only after the frozen held-out evaluation:

1. **Scientific object.** Define and estimate the
   compression–reliability frontier and the minimum retained cost
   \(b^*(\rho)\) for decision-preserving retrieval evaluation.
2. **Empirical map.** Measure complete frontiers across six heterogeneous
   retrieval datasets and show how the required budget varies by benchmark,
   reliability target, and decision margin.
3. **Method.** Introduce AutoCompress-IR, which converts a benchmark, a
   reference run panel, and a reliability target into nested weighted query
   policies plus a conservative estimate of the minimum required budget.
4. **Theory.** Characterize exact weighted compression on a finite panel,
   provide finite-population random-sampling guarantees, and derive
   margin-dependent limits on how small a decision-preserving benchmark can
   be.
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
3. optional retriever group metadata for stratified folds or Priority-1 stress
   tests;
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
- a worst-dataset or distributionally robust term across benchmark strata;
- penalties for instability across held-out-system folds; and
- explicit dataset-coverage and nesting constraints.

Exact pairwise agreement remains the evaluation metric. The smoother objective
is only an optimization device.

### First implementation

Build the first credible version before exploring elaborate search:

1. Construct the per-query utility and pair-difference matrices.
2. Create pre-declared balanced random-system cross-validation folds.
3. Learn continuous nonnegative weights with a conditional-gradient
   (Frank–Wolfe) or projected sparse optimization step.
4. Convert the support to a single priority ordering and deterministically
   round it at all budgets.
5. Improve rounded prefixes with budget-aware add/swap moves against the
   worst held-out-development fold.
6. Select hyperparameters only by nested cross-validation.
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

The Priority-0 theory should bracket how small a benchmark can be and justify
how a target-reliability budget is selected from benchmark structure, decision
margins, and held-out reliability.

### T1. Exact fitting of a finite system panel

For \(K\) known retrievers, let

\[
v_q=(x_{q,r_1},\ldots,x_{q,r_K})\in[0,1]^K.
\]

The full score vector is the mean of the \(v_q\)'s and therefore lies in their
convex hull. By Carathéodory's theorem, there exists a nonnegative weighted
subset of at most \(K+1\) queries that exactly reconstructs all \(K\) aggregate
scores.

This gives an exact finite-panel upper bound and explains why weighted
compression can be surprisingly strong when the number of evaluated systems
is small. It also shows why in-sample fit alone cannot estimate reliability.
State all qualifications:

- it permits nonuniform weights;
- it guarantees only the systems included in the finite panel;
- multiple separately constrained datasets or metrics increase the effective
  dimension; and
- finding a useful nested or uniform subset is a different problem.

### T2. Finite-population random-sampling guarantee

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

### T3. Margin-dependent limits: how micro can evaluation be?

For a protected system pair with full-benchmark margin \(\gamma\) and
per-query difference variance \(\sigma^2\), sign recovery should require on
the order of

\[
\frac{\sigma^2}{\gamma^2}\log\frac{1}{\delta}
\]

informative query observations in the corresponding statistical model, capped
by the finite benchmark size. Protecting many pairs adds a complexity term.
Prove matching upper and minimax lower bounds under explicit assumptions,
rather than presenting this heuristic rate as a theorem before it is checked.

Combine three answers to “how micro can it be?”:

1. the Carathéodory exact-fit upper bound for a fixed weighted panel;
2. distribution-free upper and lower sample-complexity bounds when only bounded
   outcomes and margins are assumed; and
3. an instance-specific optimum or lower bound from a mixed-integer solver on
   datasets small enough to solve exactly.

The empirical section should test whether query-level variance, effective
behavioral rank, and pairwise margins explain the large differences in
\(b^*(0.95)\) across datasets.

### T4. Confidence for a selected frontier point

AutoCompress-IR selects both a policy and the first tested budget that appears
to reach \(\rho\). Pointwise intervals can become optimistic after this
selection. Construct a simultaneous lower confidence band over the declared
method–budget grid using system-level resampling or a bounded U-statistic
argument, with assumptions stated explicitly. Then choose
\(\widehat b^*_{\mathcal A,\mathrm{LCB}}(\rho)\) from that band.

Validate coverage in simulation and report how often the chosen policy reaches
its target on the frozen held-out systems. This connects the formal frontier
to the practical output of the algorithm.

### T5. Optional system-panel transfer theory (Priority 1)

Two results may support the structured-holdout stress test, but they are not
required for the headline paper:

1. **Unrestricted-panel no-free-lunch.** For any strict query subset,
   construct retrievers that agree on selected queries but differ on omitted
   queries enough to reverse a full-benchmark order. This only says that a
   guarantee needs a declared population or structural assumption; it does not
   predict that named retriever families will fail.
2. **Behavioral-coverage bound.** Relate held-out score error to approximation
   by the behavioral span of the construction panel, then test the bound only
   if the residual-distance diagnostic is stable and useful.

For the second result, let \(u_r\in\mathbb R^{|Q|}\) be the per-query utility
vector of a held-out retriever and \(U\) the matrix of reference utility
vectors. For full weights \(p\) and compressed weights \(w\), decompose

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

The first term is reference-panel approximation error. The second is the
held-out retriever's distance from the reference behavioral span. State the
bound with an explicit pair of compatible dual norms; the display above is
only shorthand. The algebra alone does not imply that named system groups
differ in difficulty.

Turn this into a computable diagnostic using held-out residual distance or a
regularized projection. Keep it only if it predicts actual error beyond simple
margin and variance baselines.

### T6. Optional optimization hardness (Priority 2)

If a clean reduction is available, show that minimum-support uniform or nested
decision-preserving selection is NP-hard, motivating approximation algorithms.
This is optional. Do not spend the theory budget on it before T1–T4 are
complete and checked.

### Theory definition of done

- Every theorem has assumptions matching an experiment.
- T1–T4 have independent proof checks and synthetic sanity tests.
- T2 and T3 give constants or finite-sample calculations that can be compared
  with the empirical frontiers.
- T4 has a target-coverage simulation and a frozen-holdout calibration result.
- T5 appears only if the Priority-1 stress test earns space.
- No asymptotic result is presented as a practical saving without constants.

## Empirical design

### Public benchmark suite

The extended suite contains all English BEIR entries with at most 600,000
documents. The six Priority-0 entries are marked as core:

| BEIR entry | Documents | Test queries | Role |
|---|---:|---:|---|
| TREC-COVID | 171,332 | 50 | core; very small query set; biomedical |
| NFCorpus | 3,633 | 323 | core; small corpus; biomedical |
| FiQA-2018 | 57,638 | 648 | core; finance |
| ArguAna | 8,674 | 1,406 | core; argument retrieval |
| Touche-2020 | 382,545 | 49 | Priority 1; argument retrieval |
| CQADupStack | 457,199 | 13,145 | Priority 1; twelve heterogeneous CQA tasks |
| Quora | 522,931 | 10,000 | core; duplicate-question retrieval |
| SciDocs | 25,657 | 1,000 | Priority 1; scientific document links |
| SciFact | 5,183 | 300 | core; scientific claim evidence |
| **Total** | **1,634,792** | **26,921** | before deduplication across entries |

Verify these counts against the pinned BEIR/ir_datasets revision and preserve
the twelve CQADupStack task boundaries. Do not silently change the suite after
seeing results.

Natural Questions is a Priority-2 large-corpus stress test. It must not control
method selection or delay the core suite. Its purpose is to show how query
reduction interacts with a roughly 2.7-million-document index and therefore
with fixed versus variable cost.

The primary metric is nDCG@10. Recall@100 and MRR@10 are secondary where
defined. Report each dataset separately and an equal-dataset macro average;
never let Quora or CQADupStack dominate merely because they have more queries.

### Retriever population

The paper evaluates a fixed panel of existing, public retrieval systems.
“Retriever family” is not a single objective label, so family names are not
used to define the Priority-0 result. Record the following overlapping metadata
for panel auditing and optional stress tests:

1. retrieval mechanism: lexical, learned sparse, dense, late interaction,
   hybrid, or reranking;
2. checkpoint lineage: models sharing architecture and training ancestry;
3. supervision/training data: unsupervised, MS MARCO supervised, contrastive,
   distillation, or instruction tuned;
4. pipeline composition: first-stage model, fusion, candidate depth, and
   reranker; and
5. release time.

Freeze this metadata before outcome analysis. For a Priority-1 grouped
holdout, all checkpoints from one lineage must stay in the same split.

The minimum headline panel is roughly 16 development systems plus 4 frozen
held-out systems. Expand to 20–24 development and 4–6 held-out systems only
after the core figure is secure. The following is a candidate registry, not
permission to substitute whichever systems give the best story:

| Mechanism | Candidate systems |
|---|---|
| Lexical | BM25; BM25+RM3 |
| Learned sparse | uniCOIL; SPLADE++; one independently trained SPLADE lineage |
| Task-trained dense bi-encoder | DPR; ANCE; TAS-B |
| General-purpose dense bi-encoder | Contriever; E5 base/large; BGE base/large; GTE base/large |
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

Do not use proprietary or API-only retrievers in the primary claims. Frozen
held-out means withheld from compressor development, not unavailable to
readers. All held-out run files and manifests are released after final
evaluation.

Generating this panel is the expensive part of the project, but it happens
once. Cap the panel rather than chasing every leaderboard model. If compute is
tight, preserve broad retrieval-behavior coverage before preserving raw model
count. This is panel design, not a claim that mechanism holdout must reduce
reliability.

### System splits and optional structured stress tests

Use simple held-out-system resampling for the headline result. Group labels
support secondary stress tests; they are not ordered from “easy” to “hard,” and
no degradation is guaranteed.

| Regime | Priority | What is held out | Purpose |
|---|---|---|---|
| Balanced random-system folds | P0 | pre-declared subsets of the development panel | estimate ordinary out-of-system reliability and method variance |
| Frozen final holdout | P0 | four reproducibly selected public systems | one non-adaptive confirmation after choices are fixed |
| Checkpoint group | P1 | size, seed, or nearby checkpoints together | sensitivity to correlated panel members |
| Lineage group | P1 | all checkpoints assigned to one declared lineage | sensitivity to training ancestry |
| Mechanism group | P1 | one sparse, dense, late-interaction, hybrid, or reranking group | sensitivity to retrieval mechanism |
| Pipeline group | P1 | a fusion or reranking composition | sensitivity to pipeline composition |
| Temporal group | P2 | release-date cohorts | exploratory time-based sensitivity |

Use repeated balanced random-system cross-validation within the development
panel. Freeze 4 systems for the final Priority-0 evaluation, increasing to at
most 6 only if resources allow. Hold their outcome matrix until the method,
hyperparameters, plots, and claim thresholds are committed. Unlock it once,
archive the unlock record, and then release it publicly.

Run lineage or mechanism group holdouts only after the headline frontiers are
complete. Report a null result plainly. These systems are established models;
avoid “new model family” language unless a genuinely prospective cohort is
later collected.

### Budgets

The Priority-0 retained variable-cost budgets are:

\[
\{1,2,5,10,20,40,100\}\%.
\]

Add 60% and 80% points only if they are cheap to generate or needed to resolve
the 95%-reliability crossing.

For equal-cost analysis, use the exact integer count implied by each dataset
and report rounding. Very small test sets will make the 1% and 2% points
intentionally unstable. For measured-cost analysis, select under profiled
per-query costs.

Policies should be nested so work at a smaller budget is reused at a larger
one. Also run independently optimized non-nested policies as an upper-bound
ablation.

### Baseline pool and reporting priority

Every baseline receives the same development systems and no frozen held-out
outcomes.
Priority 0 is:

1. uniform and stratified random sampling, with at least 100 draws per budget;
2. Anchor Points and tinyBenchmarks;
3. development-panel aggregate-mean matching;
4. supervised greedy pairwise-margin preservation; and
5. AutoCompress-IR.

Choose the “strongest prior method” shown in Figure 1 using the development
folds, then freeze that choice before held-out evaluation. Keep the other
Priority-0 methods in the appendix.

Priority 1 is query-feature k-medoids, applicable methods from *A Few Good
Topics*, uncertainty-aware and intelligent topic selection, and recent
ranking-preserving dataset-selection or coreset methods whose code and cost
models can be reproduced.

A test-panel oracle may be used only to show headroom. Label it invalid for
deployment and never include it in the empirical frontier.

Active sampling methods designed to reduce relevance judging should be
included only in the related-work comparison or adapted transparently; do not
pretend their cost model is identical.

### Primary and secondary outcomes

Primary:

- held-out pairwise decision agreement under nDCG@10;
- required budget \(b^*(0.95)\) for 95% reliability; and
- equal-dataset macro and per-dataset frontiers, including worst-dataset
  reliability.

Secondary:

- aggregate-score mean and maximum absolute error;
- Kendall's \(\tau\) and Spearman rank correlation;
- winner and top-\(k\) membership agreement;
- reliability by full-benchmark pair margin;
- calibration of predicted reliability and the selected
  \(\widehat b^*_{\mathcal A,\mathrm{LCB}}(0.95)\) on frozen held-out systems;
- support overlap and policy stability across seeds;
- wall-clock, GPU-hours, energy if available, and monetary cost; and
- storage and one-time indexing cost.

Report the entire frontier. A single favorable budget is not sufficient.

### Experiment matrix

| ID | Priority | Experiment | Decision enabled |
|---|---|---|---|
| E0 | P0 | Build and audit core full run files | establishes the full-evaluation target |
| E1 | P0 | Estimate random and prior-method frontiers across six datasets | measures baseline compressibility and benchmark heterogeneity |
| E2 | P0 | AutoCompress-IR versus required baselines | tests fixed-cost reliability and fixed-reliability budget gains |
| E3 | P0 | Frontier confidence and target-budget calibration | tests whether the selected budget actually reaches its target |
| E4 | P0 | Small-instance exact solver and theory diagnostics | compares empirical compression with finite-panel and margin limits |
| E5 | P0 | One-time frozen-system evaluation | supplies the final non-adaptive confirmation |
| E6 | P0 | Targeted measured-cost study | translates nominal compression into practical savings |
| E7 | P1 | Checkpoint, lineage, mechanism, and pipeline holdouts | tests sensitivity to structured panel composition without assuming failure |
| E8 | P1 | Vary panel size/diversity and test behavioral distance | studies when broader reference coverage helps |
| E9 | P2 | Temporal, progressive-evaluation, and NQ extensions | tests optional longer-range applications |

### Required ablations

Priority 0:

- uniform versus learned weights;
- average-risk versus worst-dataset optimization;
- pairwise-decision loss versus score-reconstruction loss;
- practical tie-margin choices; and
- pointwise versus simultaneous-confidence budget selection.

Priority 1:

- nested versus independently optimized policies;
- outcome vectors versus query features versus both;
- with and without retriever-family metadata;
- random folds versus checkpoint-, lineage-, and mechanism-grouped folds;
- number of reference systems at fixed mechanism diversity;
- mechanism diversity at fixed reference-system count;
- per-dataset compression versus global suite-level budget allocation.

Priority 2:

- greedy/local search versus optional outer-loop search at matched candidate
  evaluation count.

### Statistical protocol

- Pre-register the primary metric, system splits, budgets, tie-margin
  sensitivity, and go/no-go thresholds before frozen evaluation.
- Use identical folds and random seeds for paired method comparisons.
- Resample at the retriever-system and dataset level. Retriever pairs sharing
  a system are not independent observations; do not bootstrap them as if they
  were.
- Report 95% confidence intervals and raw per-dataset values.
- Use at least 100 random-subset replicates and at least 10 method seeds where
  the algorithm is stochastic.
- Plot raw frontier estimates. An isotonic curve may summarize the expected
  monotone trend, but it must not replace raw values.
- Report failures and variance at tiny budgets, not only means.
- Use simultaneous confidence bands when selecting \(b^*(\rho)\) from several
  budgets or methods; report the point estimate separately.
- Choose hyperparameters through nested validation; never through frozen
  held-out systems.
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

Define an eight-system audit panel spanning BM25, learned sparse, task-trained
dense, general-purpose dense, instruction-conditioned, late interaction,
hybrid, and reranking mechanisms. Provide scripts to recreate its indexes and
runs on all main datasets. The remaining systems are reproducible from pinned
open models but need not be rerun for an ordinary paper reproduction.

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

1. treating **held-out decision reliability at a stated cost** as the central
   quantity;
2. mapping full compression–reliability frontiers and their inverse
   \(b^*(\rho)\), rather than reporting one subset at an arbitrary ratio;
3. establishing how frontiers vary across heterogeneous retrieval benchmarks;
4. automatically constructing nested policies and a conservative budget for a
   declared reliability target;
5. connecting exact finite-panel compression, random-sampling guarantees, and
   margin-dependent lower limits to the measured frontiers; and
6. releasing a broad, static, reproducible run panel.

For an ICLR submission, the paper needs all of the following:

- a material, reproducible variation in the compression–reliability tradeoff;
- an algorithmic gain at fixed reliability or fixed cost;
- theory that brackets or predicts the observed compressibility;
- breadth across datasets and retrieval mechanisms; and
- a reusable artifact whose claims can be reproduced cheaply.

The frontier definition alone is not enough for ICLR; without algorithmic gain
or unusually strong limits theory, this is likely an incremental IR
topic-selection paper. Structured family holdouts can strengthen external
validity, but they are not an acceptance gate. Without either a convincing
method contribution or a genuinely informative theory/empirical atlas, stop
or redirect to a smaller IR venue.

The connection to outer-loop research is practical, not rhetorical: reliable
cheap evaluation allows a model-development loop to compare more candidates
without silently changing its objective. No agent-specific assumption is
needed for the contribution.

## Proposed abstract

> Retrieval benchmarks are repeatedly used to choose among sparse, dense,
> late-interaction, hybrid, and reranking systems, yet running every query for
> every candidate can dominate repeated evaluation cost. Existing
> benchmark-compression methods typically choose a subset for a prespecified
> size; they do not answer how much compression is safe for a required level of
> decision fidelity. We formalize this tradeoff as the
> **compression–reliability frontier**: the highest held-out agreement with
> full-benchmark system comparisons attained at each retained-cost budget for
> a declared benchmark, metric, and retriever population. We introduce
> **AutoCompress-IR**, which produces nested, cost-aware query policies and
> selects the smallest tested budget whose reliability lower bound reaches a
> target. Our analysis characterizes exact weighted compression for a finite
> system panel and gives finite-population, margin-dependent bounds on the
> queries needed to preserve pairwise decisions. Across [DATASETS] and
> [SYSTEMS], the budget required for [RELIABILITY] reliability ranges from
> [LOWEST BUDGET] to [HIGHEST BUDGET], showing [HETEROGENEITY RESULT]. At the
> same target, AutoCompress-IR reduces retained cost by [METHOD RESULT] versus
> [STRONGEST BASELINE]. We release policies, static run files, and code that
> reproduces every frontier without rerunning retrieval models.

All bracketed fields are mandatory result placeholders. Do not replace them
with qualitative claims until the frozen held-out run is complete.

## Paper outline

1. **Introduction:** evaluation cost, the danger of a universal tiny
   benchmark, and the frontier question.
2. **Problem formulation:** ranked-run abstraction, cost, decisions, fidelity,
   reliability, the frontier, and \(b^*(\rho)\).
3. **How micro can evaluation be?:** exact finite-panel compression,
   random-sampling guarantees, margin-dependent limits, and frontier
   confidence.
4. **AutoCompress-IR:** behavioral representation, nested cost-aware selection,
   and reliability calibration.
5. **Experimental protocol:** BEIR suite, fixed retriever registry, baselines,
   held-out-system protocol, leakage controls, and costs.
6. **Results:** headline frontier, per-dataset required budgets,
   AutoCompress-IR comparison, theory diagnostics, and frozen-holdout
   calibration.
7. **Practical use:** target-reliability policy selection and measured savings.
8. **Related work and limitations:** IR topic selection, benchmark
   compression, qrel limitations, system-population scope, and environmental
   cost.
9. **Conclusion:** compression is a conditional reliability decision, not a
   fixed subset.

## Execution plan

### Phase 0 — Freeze the question and protocol

- [ ] Copy the definitions in this document into protocol.md.
- [ ] Confirm the official ICLR 2027 timeline and work backward from it.
- [ ] Complete a claim-by-claim related-work table.
- [ ] Verify all dataset counts and licenses.
- [ ] Define the exact pairwise decision rule and tie-margin sensitivity.
- [ ] Pre-register budgets, held-out-system folds, confidence intervals, and
      pilot gates.
- [ ] Create datasets.lock.yaml and an initial systems.lock.yaml.
- [ ] Freeze code/data boundaries so final-holdout outcome files cannot be
      loaded by development jobs.

**Deliverable:** a versioned protocol whose primary analysis cannot be changed
after seeing final held-out results.

### Phase 1 — Build the static evaluation panel

- [ ] Implement one run adapter and per-query metric schema.
- [ ] Smoke-test every candidate system on TREC-COVID and SciFact.
- [ ] Select roughly 16 development and 4 frozen held-out open systems for
      Priority 0.
- [ ] Generate or acquire full ranked runs for the six core BEIR entries.
- [ ] Expand systems and datasets only after the headline experiment works.
- [ ] Preserve CQADupStack constituent-task boundaries if it is added.
- [ ] Profile fixed and per-query costs for every mechanism.
- [ ] Validate checksums, score direction, query counts, and metric parity.
- [ ] Implement the eight-system end-to-end audit path.

**Deliverable:** immutable run and per-query outcome matrices plus cost and
panel metadata.

### Phase 2 — Run the decisive pilot

- [ ] Use all six core datasets: TREC-COVID, NFCorpus, FiQA, ArguAna, Quora,
      and SciFact.
- [ ] Use at least 12 reproducible systems spanning the main retrieval
      mechanisms.
- [ ] Run random, stratified random, visible mean matching, Anchor Points,
      tinyBenchmarks, greedy pairwise preservation, and AutoCompress-IR.
- [ ] Measure 1%, 2%, 5%, 10%, and 20% budgets.
- [ ] Use repeated balanced random-system folds; reserve structured holdouts
      for Priority 1.
- [ ] Plot all three headline method curves and per-dataset
      \(b^*(0.95)\).
- [ ] Estimate simultaneous confidence bands and check whether the first
      selected target budget is calibrated in resampling simulations.
- [ ] Run the go/no-go review below before scaling.

**Deliverable:** one compact evidence packet that decides whether the ICLR
story exists.

### Phase 3 — Complete AutoCompress-IR

- [ ] Implement behavioral difference vectors and balanced system folds.
- [ ] Implement continuous sparse weighting and deterministic rounding.
- [ ] Implement budget-aware local swaps and nesting.
- [ ] Add worst-dataset optimization and reliability calibration.
- [ ] Add per-dataset coverage constraints and cost-aware selection.
- [ ] Build a small-instance mixed-integer oracle.
- [ ] Run every required ablation.
- [ ] Freeze method and hyperparameters before final held-out evaluation.

**Deliverable:** a documented command that accepts benchmark artifacts and
emits policies plus a frontier report.

### Phase 4 — Prove and test the theory

- [ ] Write formal statements and proofs for T1–T4.
- [ ] Check edge cases: weights, multiple metrics, strata, ties, and finite
      populations.
- [ ] Derive finite-sample constants for random sampling and margin-dependent
      upper and lower limits.
- [ ] Build synthetic examples spanning easy, redundant, high-variance, and
      near-tie regimes.
- [ ] Compare theory with a mixed-integer optimum or bound on small datasets.
- [ ] Validate simultaneous-band coverage and selected-budget calibration in
      simulation.
- [ ] Seek an independent proof review.
- [ ] Attempt structured-panel transfer theory only after the core theory is
      complete and a Priority-1 experiment motivates it.

**Deliverable:** theory that explains a measured phenomenon and survives an
independent check.

### Phase 5 — Full evaluation and one-time held-out check

- [ ] Run all Priority-0 methods, budgets, and datasets using frozen
      random-system folds.
- [ ] Generate the macro frontier, six per-dataset frontiers, and
      \(b^*(0.95)\) comparison.
- [ ] Complete the two targeted measured-cost studies.
- [ ] Commit the final code, method choice, and analysis notebook.
- [ ] Unlock the frozen held-out panel exactly once.
- [ ] Run the pre-registered held-out analysis without method changes.
- [ ] Release held-out run files and record the unlock.
- [ ] If a bug requires rerunning, document it and label the held-out panel
      compromised rather than silently resetting it.

**Deliverable:** final numbers with a complete provenance trail.

### Phase 6 — Write and release

- [ ] Fill abstract placeholders only from the frozen result tables.
- [ ] Lead the paper with the frontier, benchmark heterogeneity, and the
      fixed-reliability method comparison.
- [ ] Put every per-dataset result and all Priority-1 structured-holdout results
      in the appendix/artifact.
- [ ] Include limitations about the declared system population, qrels, corpus
      indexing, and full-benchmark uncertainty.
- [ ] Package the CPU-only reproduction path.
- [ ] Reproduce every figure and table from a clean environment.
- [ ] Run an internal skeptical review against the ICLR bar.

**Deliverable:** submission, public artifacts, and an auditable result ledger.

## Pilot go/no-go criteria

Set these thresholds before running the pilot to avoid rationalizing a weak
signal.

### Gate A: the frontier is informative and practically nontrivial

Pass if the pilot can estimate stable reliability curves and supports both of
these statements:

- at least one legitimate method reaches a 95% reliability lower confidence
  bound at 20% retained query cost or less on at least four of the six core
  datasets; and
- the required budget differs by at least twofold across core datasets, with
  uncertainty or sensitivity analysis ruling out the claim that one fixed
  ratio is adequate everywhere.

Also inspect results by pairwise margin so the effect does not depend on a
single pathological system pair. If the exact numerical threshold proves
statistically ill-posed in the pilot, revise and commit it before any frozen
held-out result is opened.

### Gate B: the proposed method adds value

Pass if AutoCompress-IR, against the strongest legitimate baseline, either:

- improves held-out pairwise agreement by at least 2 absolute points at a
  budget of 10% or less; or
- reduces the budget required for 95% reliability by at least a factor of 1.5,

without losing more than one point on worst-dataset reliability. Confirm the
same direction on most main datasets in the full study.

### Gate C: the theory says something non-vacuous

Pass if the finite-population and margin-dependent results yield numerical
bounds or difficulty predictions that can be compared with the empirical
frontiers, and if selected-budget confidence reaches its nominal coverage in
simulation. A restatement of Carathéodory plus asymptotic big-O notation is not
enough for the ICLR claim.

### Gate D: the savings are operationally meaningful

Pass if a 10% query policy yields at least a fivefold measured speedup for an
index-reuse or reranking workload and the paper can name a realistic repeated
evaluation setting where fixed costs do not erase the benefit. Report the
fresh-index case even if it shows little end-to-end saving.

### Gate E: the result is practically reproducible

Pass if the full run panel can be frozen, the central analyses regenerate from
static artifacts on CPU, and the audit panel can be rebuilt from pinned public
models with matching run checksums or documented numerical tolerance.

### Decision

- **A + B + C + D + E pass:** proceed as the intended ICLR 2027 paper.
- **A passes, B fails:** consider a frontier/limits paper only if the theory and
  benchmark atlas are unusually informative; otherwise target an IR venue.
- **B passes, A fails:** this may still be a compression-method paper, but the
  “no universal ratio” claim must be narrowed to what the data establish.
- **C fails:** the work needs a stronger algorithmic or empirical contribution
  to clear the ICLR bar.
- **D fails:** retain the statistical contribution only if it is strong, and
  remove broad compute-saving claims.
- **A or E fails and no strong replacement result appears:** stop or narrow
  the project.

Results from lineage, mechanism, pipeline, or temporal holdouts are not a
go/no-go gate. Promote them only if they reveal a large, reproducible secondary
finding after the Priority-0 story is complete.

The numerical gates are planning thresholds, not claims of statistical or
practical significance for every use case. Freeze or revise them, with written
reasoning, before looking at pilot outcomes.

## Handoff rules for the executing agent

1. Start with Phase 0 and the pilot. Do not launch the full matrix before the
   go/no-go review.
2. Maintain a result ledger with one row per sentence-level claim, its script,
   artifact checksum, table/figure, and status.
3. Never tune on frozen held-out outcomes, including by manually viewing their
   summary statistics.
4. Keep failed methods and negative datasets in the artifact.
5. Distinguish exploratory plots from frozen confirmatory analyses.
6. Record every panel or protocol change in Git before rerunning.
7. Prefer public run artifacts and deterministic pipelines over a larger but
   irreproducible system count.
8. Keep family, lineage, and mechanism holdouts at Priority 1. Do not call the
   fixed public systems “new,” and do not imply that structured holdouts must
   be harder.
9. Do not claim a global optimum. Report an empirical method frontier and an
   oracle only where its information access is explicit.
10. Keep the main message crisp: **each retrieval benchmark has a
    compression–reliability frontier, and AutoCompress-IR finds the smallest
    tested evaluation that meets a chosen reliability target.**

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
out at all, optional lineage/mechanism/time grouping, theory, and released
artifacts. The novelty section should be rewritten from that matrix rather
than from memory.
