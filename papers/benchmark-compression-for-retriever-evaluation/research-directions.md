# Stronger Research Directions for IR and Self-Improving Search

Research proposals and comparative ratings · 8 September 2026

Related documents: [original frontier proposal](README.md) ·
[frontier novelty and validity assessment](frontier-assessment.md).

My first research bet is an agent that learns how to search an unfamiliar environment through a small number of self-generated experiments. My more concrete method bet is retrieval that learns which documents help the next search step.

**Choose a learning problem with a falsifiable mechanism before choosing the paper outline.** The strongest continuation of the original motivation is transferable self-improvement: what does an agent learn from interacting with a search environment, and does that learning help on new tasks? Benchmark compression can remain experimental infrastructure.

These are proposals, not established novelty claims or acceptance predictions. The literature check rules out several broad pitches; it does not establish that every narrower idea is untouched. A successful method can still make a substantial contribution to a problem that prior work already identifies.

Assumptions: public data, modest initial compute, flexible timeline. No training or evaluation experiments were run. Literature was checked through September 8, 2026; several close papers are recent preprints. Their reported results have not been reproduced here.

<a id="choice"></a>

## Which bet would I choose?

| Direction | Why pursue it | Main uncertainty |
| --- | --- | --- |
| [1. Learn an unfamiliar search environment](#explore) | My lead for an ambitious ICLR-oriented project. Direct connection to autonomous improvement and adaptation. | Can deliberate exploration beat clear instructions, passive experience, and synthetic-query adaptation? |
| [2. Learn the future search value of evidence](#value) | Clearest method project and a natural IR contribution. Start with a small oracle experiment. | Does an affordable predictor improve decisions and transfer beyond its training agent? |
| [3. Study evaluation as a limit on self-improvement](#feedback) | Best reuse of current assets and cheapest path to an informative pilot. | Is there a consequential mechanism beyond ordinary adaptive overfitting? |
| [4. Learn compositional evidence applicability](#scope) | Strong practical motivation in technical search; potentially broader compositional learning. | Does a hard residual problem remain after filtering and prompted verification? |

I would pilot **1 and 2** before committing to a larger project. If minimizing new infrastructure is the dominant constraint, pilot **3** first. These are separate paper candidates, not four work packages to combine into one submission.

### Novelty, lightbulb potential, and engineering / experimental lift

These are qualitative judgments on a **1–5 scale**, based on the ideas as
currently specified and the literature reviewed above. They are not measured
results, acceptance probabilities, or scores to average into a single ranking.

- **Novelty today:** 1 = established framing; 2 = close precedent with a narrow,
  provisional gap; 3 = a plausibly distinct research target or mechanism;
  4 = a clearly differentiated method or finding; 5 = a new research paradigm.
  Ratings can rise or fall after a focused prior-art check and pilot.
- **Lightbulb potential:** 1 = routine optimization; 2 = useful refinement;
  3 = a meaningful method insight; 4 = a result that could change how we understand
  learning or retrieval; 5 = a field-shaping principle. This rates the upside of
  a successful result, not how likely that result is.
- **Engineering / experimental lift:** 1 = mostly existing artifacts and light
  analysis; 2 = a small new prototype and bounded evaluation; 3 = substantial
  integration, repeated experiments, or modest training; 4 = major data curation
  or multiple environments/models plus extensive controls; 5 = large-scale
  infrastructure and training. Higher means harder. **Pilot → full study**
  separates the first decisive experiment from a convincing submission.

| Direction | Novelty today | Lightbulb potential | Engineering / experimental lift: pilot → full study |
| --- | --- | --- | --- |
| [1. Learn an unfamiliar search environment](#explore) | **3 / 5** | **4 / 5** | **2 → 4 / 5** |
| [2. Learn the future search value of evidence](#value) | **2 / 5** | **3 / 5** | **2 → 3 / 5** |
| [3. Study evaluation as a limit on self-improvement](#feedback) | **2 / 5** | **4 / 5** | **1–2 → 3 / 5** |
| [4. Learn compositional evidence applicability](#scope) | **2 / 5** | **4 / 5** | **2 → 4 / 5** |
| [Reserve: selective index repair](#parked) | **2 / 5** | **3 / 5** | **2 → 4 / 5** |
| [Original frontier proposal, for comparison](README.md) | **1 / 5** | **2 / 5** | **2 → 4 / 5** |

**Why these ratings:**

- **Unfamiliar-environment exploration** has the strongest potential distinction:
  learning which experiments acquire reusable search knowledge. Synthetic tasks
  and procedural memory already exist. The full study needs varied environments,
  a learned exploration policy, transfer controls, and honest amortized costs.
- **Future search value** has a clear method opportunity, but Bridge Evidence
  already identifies the problem and suggests learning its predictor. A useful
  solution can still be publishable. Branched agent rollouts and cross-agent
  evaluation drive the lift.
- **Evaluation feedback** could reveal why an apparently reliable evaluator
  limits automated discovery, which gives it high conceptual upside. Adaptive
  overfitting and reusable holdouts are established, so the current novelty is
  low until a new predictive mechanism emerges. The pilot is relatively cheap
  if suitable cached ranked runs exist; reduced-corpus matrices alone only
  support protocol debugging.
- **Compositional applicability** could explain a general failure to combine
  conditions across evidence. Existing scope and temporal-validity work sharply
  narrows the gap. Auditing real conditional cases, separating retrieval from
  reading errors, and testing unseen combinations make the full study demanding.
- **Selective index repair** faces both partial-backfilling and representation-
  capacity precedents. A new constraint-guided repair method is only a hypothesis;
  a full study needs index updates, strong adaptation baselines, and cost accounting.
- **The original frontier** is a useful organizing view, but the curve,
  target-budget inversion, and held-out-system framing have direct precedents.
  Its proposed full-corpus, multi-retriever study is still expensive. The rating
  applies to that outline, not to every possible new method for reliable evaluation.

No direction receives a 5: none yet has a demonstrated new mechanism or result.
The generic memory, sufficiency, and grounding ideas discussed in reserve are
excluded as standalone proposals; their broad framing is already established.

<a id="explore"></a>

## 1. Can a search agent teach itself how to use a new corpus?

**Research question:** Can an agent use a bounded set of self-generated search experiments to learn a procedure that improves future tasks in an unfamiliar environment, without human relevance labels or answers to those tasks?

A search policy that works on Wikipedia may struggle with API documentation, scientific abstracts, or an unfamiliar enterprise index. The useful query forms, aliases, metadata fields, and routes between documents differ. The opportunity is to learn these properties before many user tasks arrive.

**Hypothesis.** Carefully chosen probes reveal reusable properties of the environment more efficiently than accumulating ordinary task trajectories. The benefit survives changes in evaluation entities and answers, and pays back the exploration cost over a realistic number of later tasks.

### A concrete mechanism to investigate

Let the agent sample documents and create rediscovery tasks: find a known document using an indirect description, an alias, a relation, or a conjunction of conditions. Document identity supplies self-supervision. Paired probes can distinguish hypotheses about which query transformations work. A small learned controller chooses the next informative probe and updates a compact search procedure.

The eventual learning contribution could be a policy for *choosing experiments*, trained across source environments and adapted on unseen ones. A first pilot can use prompt or memory updates. It should not require large-scale RL to establish that useful information exists.

<details>
<summary>Closest work and the proposed distinction</summary>

[Search-R1](https://arxiv.org/html/2503.09516) already trains search behavior with outcome rewards. [APEx](https://arxiv.org/html/2609.02253) already combines procedural skill distillation and test-time RL. [DocArena](https://arxiv.org/html/2606.26122) builds corpus-grounded training environments. [GrepSeek](https://arxiv.org/html/2605.29307) learns direct corpus interaction, and [SearchWiki](https://arxiv.org/html/2608.29953) builds and navigates a knowledge structure.

The proposed distinction is **deliberate, bounded adaptation to an unseen environment without target-task labels**, with evidence that the learned procedure transfers independently of answer storage. Memory, synthetic queries, and self-generated tasks are established ingredients. Classical synthetic-query retriever adaptation and broader unsupervised/meta-RL require a further focused novelty audit if this pilot succeeds.

</details>

### Smallest useful pilot

Use two or three public collections, a fixed search agent, and lexical/dense interfaces. Compare no adaptation, an accurate interface description, random probes, passive experience, deliberate probes, and corpus-generated synthetic-query adaptation at matched cost. Illustrative exploration budgets are 0, 10, 50, and 200 probes; these are experimental choices, not forecasts.

Keep probe and test answers/entities disjoint. Include at least one natural change in corpus or index behavior. Measure task success, search calls, tokens, and the number of later tasks needed to repay exploration. Coherent changes to corpus facts can provide an additional test that the improvement concerns procedure.

**Stop if** a short tool description removes the gain, improvement depends on answer overlap, or random/passive probes are equally effective. **Advance if** active exploration produces portable improvements at a useful amortized cost.

**What could make it ICLR-level:** an effective learning rule for acquiring environment knowledge, robust transfer, and an explanation of when exploration helps. A small prompt-memory gain in one search interface would be insufficient. This is a promising connection to self-improvement; it is not evidence of recursive improvement in learning ability.

<a id="value"></a>

## 2. Retrieve evidence for its effect on the next search

**Research question:** Can a retriever learn a document’s contribution to future evidence acquisition, conditioned on what the agent already knows?

A document may reveal an obscure project name that unlocks the decisive next query. After that name is known, the same document adds little. Ordinary relevance ranking does not directly optimize this changing value.

**Hypothesis.** A cheap predictor of this future search value improves end-to-end success under a fixed evidence budget, beyond relevance, diversity, and simple entity-novelty heuristics.

### A concrete method

Start with a fixed agent. Cache search prefixes, substitute candidate documents, and continue the agent for a small number of paired rollouts. Use the resulting success and cost differences to train a small state–document value model. Rerank candidates using that value together with relevance. If the oracle experiment supports the idea, investigate affordable label acquisition and transfer across agent policies.

<details>
<summary>Closest work and the proposed distinction</summary>

[Bridge Evidence](https://arxiv.org/html/2607.15253) already documents static-versus-trajectory utility differences and explicitly proposes cheaply predicting trajectory utility as future work. [CVT-RL](https://arxiv.org/html/2606.05263) already uses policy-conditioned counterfactual credit and learned outcomes. [Search-G1 v1](https://arxiv.org/html/2608.07531v1) uses evidence-deletion-based signals and document replacement tests.

The contribution must therefore be an **effective, affordable, transferable ranking method**. Neither the existence of bridge documents nor counterfactual credit is new. Following up on an explicit open problem is worthwhile when the solution is substantial.

</details>

### Smallest useful pilot

Use roughly 200–300 multi-hop questions and a fixed small open agent. First compare oracle trajectory-value selection with a strong relevance reranker at the same token budget. If the ceiling is material, fit a small predictor. Required controls include MMR/diversity, entity novelty, question-only versus full-state ranking, and simply retrieving more documents at the same total cost.

Use length-matched document substitutions so deleting a document does not simply shorten the context. Evaluate uncertainty at the question level. A second corpus and second agent are necessary before claiming transfer.

**Stop if** oracle gains are tiny, a simple novelty heuristic captures them, or rollout labeling costs erase the benefit. **Advance if** a cheap predictor delivers robust gains and works across policies.

**What could make it ICLR-level:** a generalizable representation or learning principle for evidence value across changing agents. A strong method on several retrieval tasks is also a natural SIGIR/ACL/EMNLP contribution; those venues are appropriate scientific targets in their own right.

<a id="feedback"></a>

## 3. When does cheap evaluation limit self-improvement?

**Research question:** Can two evaluators with similar held-out accuracy lead an adaptive optimizer to discover very different final systems, and can we predict and prevent that divergence?

The current proposal studies an evaluator against an existing model panel. In a development loop, the evaluator changes which models are produced next. An optimizer may learn modifications that exploit a reusable evaluator error. Cheap scores can then improve while independently measured quality stagnates or declines.

**Hypothesis.** How candidate changes align with structured evaluation errors predicts final discovery quality better than passive agreement and trial count alone. This may create a regime where more search is harmful, but that result must be tested rather than assumed.

### A concrete method to earn the claim

Track behavioral departures from the systems used to validate the evaluator. Use a small independent audit stream to test whether recent improvements are real, then selectively refresh the evaluator. The diagnostic must use information available during search; a full-benchmark oracle can explain failures but cannot serve as a deployable method.

<details>
<summary>Closest work and the proposed distinction</summary>

[The Ladder](https://proceedings.mlr.press/v37/blum15.html) and [adaptive data analysis / reusable holdouts](https://arxiv.org/html/1506.02629) already establish the danger of adaptive feedback and provide remedies. [How benchmark prediction from fewer data misses the mark](https://arxiv.org/html/2506.07673) already shows failure under model extrapolation. [How Reliable is Language Model Micro-Benchmarking?](https://arxiv.org/html/2510.08730v2) already studies passive decision reliability.

The proposed advance is a **predictive mechanism and better discovery outcome in a real feedback loop**. Showing that repeated noisy selection overfits, or periodically checking a holdout, would not suffice.

</details>

### Smallest useful pilot

Use cached ranked runs on two public datasets to generate new rank-fusion or query-routing candidates adaptively. Routing must use deployable query features, not query IDs or hidden judgments. A static matrix of pre-existing model rows does not demonstrate adaptive candidate creation.

Compare fixed random subsets, a learned compressor, fresh sampling, thresholded/reusable-holdout feedback, and full feedback. Match total development cost. Plot final independent quality against total cost and separate fixed-benchmark fidelity from generalization to untouched queries.

**Stop if** the phenomenon is explained by ordinary winner’s curse or fresh sampling plus a final audit solves it cheaply. **Advance if** a substantial, predictable interaction remains and the remedy improves final systems at matched cost.

**What could make it ICLR-level:** a substantive result about the dynamics of automated discovery. This is the best use of the existing compression infrastructure, but it should not be selected solely because the infrastructure already exists.

<a id="scope"></a>

## 4. Retrieve the evidence whose conditions apply

**Research question:** Can a retriever compose validity conditions across documents and generalize to new combinations of version, configuration, and exceptions?

“The feature is enabled by default” can be true for version 2, false for version 1, and overridden in a particular mode. All the documents may be correct. A query about version 1 needs a different answer from an otherwise identical query about version 2. Some exceptions are documented separately from the default.

**Hypothesis.** A learned representation of applicability, coupled with retrieval of missing conditions and exceptions, improves unseen condition combinations beyond metadata filtering and a strong prompted verifier.

<details>
<summary>Closest work and the proposed distinction</summary>

[HoH](https://aclanthology.org/2025.acl-long.301/) already studies outdated evidence. [Temporal Validity on Real Software Histories](https://arxiv.org/html/2608.20685) handles atomic supersession. [Controlled Memory Interference](https://arxiv.org/html/2608.07622) is a serious collision: it already covers scope, authority, transient changes, and applicable stable updates.

The narrower bet is **query-dependent validity among simultaneously correct branches, with held-out combinations of conditions**. Broad applicability-aware retrieval is not an adequate novelty claim. This direction needs further checking against conditional reasoning and version-aware retrieval before substantial investment.

</details>

### Smallest useful pilot

Audit 30–50 real cases from public versioned documentation before constructing a benchmark. Use matched queries that change one condition over the same corpus. Include ordinary ranking, metadata filtering, prompted verification, and explicit condition composition. Give readers oracle evidence in a diagnostic condition to distinguish retrieval from interpretation failures.

**Stop if** metadata or a short prompt handles the cases. **Advance if** realistic cross-document exceptions remain difficult and a learned method transfers to unseen combinations and product families.

**What could make it ICLR-level:** a result about compositional learning and retrieval, rather than a version-filtering feature. This is an attractive practical IR project but a higher-risk conceptual bet than its broad motivation initially suggests.

<a id="parked"></a>

## Ideas I would keep in reserve

<details>
<summary>Selective index repair: attractive, but substantial prior work</summary>

One possible core IR project is to diagnose ranking constraints that a frozen document index cannot satisfy through query adaptation, then repair only the limiting document representations. The proposed distinction would be repair guided by task-specific representational constraints.

However, [FastFill](https://arxiv.org/html/2303.04766) already learns a policy for partial backfilling using uncertainty; [Darwinian Model Upgrades](https://arxiv.org/abs/2210.06954) already uses selective compatibility. [Align Then Adapt](https://arxiv.org/html/2604.03403) and [Query Drift Compensation](https://arxiv.org/html/2506.00037) already support adaptation with frozen document indexes. [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038) already studies the representational ceiling, including freely optimized embeddings.

The cheap first experiment would measure an oracle query-vector ceiling on real tasks. If that ceiling is effectively perfect, a geometry-based repair story has little support. If the gap is substantial, test whether targeted repairs beat FastFill-style uncertainty and strong query adapters at the same re-encoding cost. The ceiling and partial-refresh framing are not themselves new. The theory paper and Darwinian paper were checked at abstract level in this pass; FastFill, ERA, and QDC received targeted full-text reading.

</details>

<details>
<summary>Generic agent memory, evidence sufficiency, and counterfactual grounding</summary>

These remain useful areas, but recent primary papers already cover much of the obvious pitch: [APEx](https://arxiv.org/html/2609.02253) for procedural memory and adaptation; [Knowing You Don’t Know](https://arxiv.org/html/2505.02811) for learned sufficiency and search continuation; [Learning Evidence Sufficiency Boundaries](https://arxiv.org/html/2609.01687) for partial-to-sufficient-to-redundant context; [Search-G1 v1](https://arxiv.org/html/2608.07531v1) for evidence-reliance rewards and replacement tests.

Coherent fact-changing corpus interventions are useful controls for proposals 1 and 2. They are a less convincing standalone paper unless they uncover a new mechanism and support a correction.

</details>

<a id="next"></a>

## The decision before the paper

**My preferred high-level proposal:** *Learning to Explore Unfamiliar Search Environments Without Task Labels.* Its motivating question is whether an agent can learn how to find evidence through interaction and carry that improvement to future tasks. It connects naturally to the original self-improvement interest while making the transferable capability explicit.

The first deliverable should be a comparison showing whether deliberate exploration adds useful, reusable information beyond strong simple baselines. In parallel, a small oracle experiment for future-search evidence value offers a concrete alternative. Pick one direction after those results; then write its statistical contract and full experimental plan.

For ICLR, I would look for three things: a learning mechanism that differs materially from its nearest papers, a substantial result at matched total cost, and transfer that survives realistic controls. I would pursue these ideas as research bets; I would not yet predict acceptance for any of them.

This memo synthesizes targeted primary-source checks with the
[frontier assessment](frontier-assessment.md) of
[outer_loop PR #1](https://github.com/yzzhu-ron/outer_loop/pull/1). The linked
papers, scope notes, baselines, and stop/go criteria above provide the context
needed to evaluate these proposals without access to a local research workspace.
