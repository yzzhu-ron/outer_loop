# What must be learned before the query is known?

**Development note, 10 September 2026.** The useful direction is to characterize which future query types a probe policy can serve, then test whether that predicted coverage transfers. The results below are proved finite-model facts and exact constructions. They are supporting theory, not a claim of a new general experimental-design theorem. [The literature audit](LITERATURE.md) identifies close precedents, including decision-region determination, reward-free exploration, EPIG and GoBOED.

## 1. First distinguish an unknown workload from an unknown decision rule

Let θ be an environment, x a query context observed at serving time, and a a retrieval action. Utilities Uθ(x,a) lie in [0,1]. Onboarding history H is collected before any future query. A source-conditioned posterior w over θ and a workload μ over x specify the finite model. For this note, θ and x are independent given the declared μ; a change of μ preserves conditional utilities and observation laws. This deliberately isolates mixture shift. It does not cover changing relevance inside a bucket, a world-dependent context distribution that reveals θ, or source models missing the target environment.

With unrestricted conditional routing, the Bayes rule a_w(x) ∈ argmax_a Σθ wθUθ(x,a) simultaneously maximizes expected utility for every μ. Proof: each context's contribution is separately maximized, and all mixture weights are nonnegative. Unknown context frequencies therefore change acquisition priorities, but do not change this conditional rule. In contrast, a single fixed action, a shared parameter restriction, or a serving-cost constraint couples contexts and can make the optimal decision depend on μ. A task-mixture impossibility proved only for fixed actions must not be stated for an unrestricted query-conditioned router.

## 2. Keep a vector of residual decision risks

Define r_x(w) = Σθwθ max_a Uθ(x,a) − max_a ΣθwθUθ(x,a). For a probe or complete budget-feasible onboarding plan q, let D_xq = E_H[r_x(w_H)] under its observation law. These are Bayes risks within the supplied model, not target confidence guarantees. For a finite uncertainty polytope M = conv{μ¹,…,μᴸ} and a randomized choice η of budget-feasible plans, the ex-ante risk is

**R_M(η) = max_{μ∈M} μᵀDη.**

Nature chooses μ before the random plan and its observations. Standard finite minimax duality gives min_η max_l (μˡ)ᵀDη = max_λ min_q Σ_l λ_l(μˡ)ᵀD_:q. Thus a least-favorable mixture of workloads is a checkable witness for the selected plan mixture. The supplied implementation solves primal and dual separately with exact rationals. It is a tiny-game verifier, not a scalable planner; columns must represent plans satisfying the same total-cost constraint. This formulation does not justify dividing a two-step gain only by its first step's cost.

For M equal to the full context simplex, a single plan q is optimal for every workload **iff** D_xq = min_j D_xj for every context x. Sufficiency is summation; necessity follows by choosing μ concentrated on each x. A pair of plans whose risks cross across contexts supplies explicit workloads that reverse their ranking. Absence of such a crossing is not, by itself, a proof that all sequential acquisition policies coincide; the check is local to the declared plan menu, prior and utility tensor.

The zero-risk version is decision-region determination: after every positive-probability transcript, every context allowed by M must have an action optimal in every surviving positive-prior world. Necessity follows because expected regret is a sum of nonnegative terms; sufficiency chooses that common action context by context. Overlapping regions and multiway conflicts are established DRD territory, not new claims here.

## 3. The order of worst case and expectation is part of the contract

For a fixed unknown workload, evaluate **max_μ E_H[μᵀr(w_H)]**, not E_H[max_μ μᵀr(w_H)]. The second expression allows the workload to react to the observed transcript. Each r_x is concave in w, so E_H r_x(w_H) ≤ r_x(w); the correct fixed-workload value cannot be negative. Moving max inside the expectation need not preserve this property.

An exact four-world example makes the issue visible. Give each world prior 1/4. The pairs of optimal binary labels at two contexts are (0,0), (1,0), (0,0), (0,1), with unit reward for a correct label. A signal identifies whether the world is in the first or last pair. Before the signal, both context risks are 1/4. Afterward they are either (1/2,0) or (0,1/2). Fixed-workload worst risk stays 1/4; the outcome-reacting expression is 1/2, producing a spurious −1/4 “VOI” for the fixed-workload problem. This illustrates a known robust-conditioning/dilation issue; it is not evidence that ordinary Bayesian information is harmful.

## 4. A sharp construction quantifies the cost of learning before task specification

There are m contexts and m independent fair bits θ₁,…,θₘ. Context i rewards guessing θᵢ. Each unit-cost probe reveals one chosen bit without noise; at most B probes are allowed, adaptively. The future context is selected before onboarding but revealed afterward. The exact minimax error is

**R*(m,B) = ½ max{1 − B/m, 0}.**

Proof: if bit i is never observed, its posterior remains fair, even under adaptive selection based on other observed bits. Write πᵢ for its probability of observation. Context-i error is at least (1−πᵢ)/2, and Σᵢπᵢ ≤ B; some context therefore has error at least (1−B/m)/2. Uniformly choose a subset of min(B,m) coordinates and guess any unobserved bit uniformly. Every inclusion probability equals min(B/m,1), attaining the bound. A query-aware probe can instead achieve zero with one probe; that is a different information contract and its repeated per-query costs must be charged.

Suppose a deterministic message about the future context, with at most K possible values, is supplied **before** onboarding. It partitions the contexts; the largest cell has at least ceil(m/K) members. Applying the preceding argument within each cell and using balanced cells gives the exact tradeoff

**R*(m,B,K) = ½ max{1 − B/ceil(m/K), 0}.**

In this construction, zero regret with B≥1 requires at least ceil(m/B) messages; communicating the relevant task family can replace exploration. This is a restricted direct-sum/counting construction, related to reward-free exploration and indexing problems. Its useful contribution is a falsifiable search hypothesis about context coverage, not an asserted new complexity theorem.

## 5. A richer menu makes the workload tradeoff observable

The executable example has two decision bits and three nuisance bits, hence 32 worlds. Two probes reveal the respective decision bits; a third reveals all three nuisance bits. Their information gains are 1, 1 and 3 bits, while their context risk columns are (0,1/2), (1/2,0), and (1/2,1/2). A source workload (0.9,0.1) chooses the first decision probe and obtains risk .05; shifting entirely to the second context raises that policy's risk to .5. Randomizing equally between the two useful probes has worst-workload risk .25, versus 1/3 for uniform random choice over all three probes. These are exact constructions, not retrieval measurements or a claim that our real probes share this structure.

This also clarifies the old binary Gaussian limitation. With two worlds and common within-probe variance, standardizing each scalar probe gives Y = dθ + Z for θ∈{0,1}. If d₁≥d₂≥0, transform Y₁ to (d₂/d₁)Y₁ + sqrt(1−(d₂/d₁)²)Z′. It has precisely Y₂'s law; zero separation is an uninformative special case. Thus the larger-separation channel Blackwell-dominates the smaller one. It weakly improves every context's Bayes decision risk and maximizes information gain at equal cost. More worlds, categorical outcomes, or crossing posterior decisions can remove this collapse, but **counting worlds is insufficient**: verify actual channel/decision tradeoffs.

## 6. What a transfer guarantee would need

Suppose the true residual-risk matrix and source prediction satisfy |D_xq − D̂_xq| ≤ ε_x uniformly over the frozen plan menu. Put h_M(ε)=max_{μ∈M} μᵀε. If η̂ minimizes the predicted robust risk, then its true robust risk is at most the true robust optimum plus **2h_M(ε)**. Proof: every plan mixture's true and predicted robust values differ by at most h_M(ε); apply this once at η̂ and once at the true optimum. If a deployment mixture is within total variation δ of some μ∈M, its risk has an additional upper allowance δ because losses lie in [0,1]. The comparator remains the robust optimum over M, not an oracle told the actual target mixture.

This is the standard optimization-perturbation bound. The scientific challenge is making ε small and measurable on held-out source families. Source-only fit error, concentrated pseudo-posteriors, and good reconstruction of probe outcomes do not establish that condition. This is exactly where the natural development results suggest the current model is weakest.

## 7. The decisive empirical use

First audit policy capacity: can any posterior in the current source model represent a competitive query router? If not, refine the utility/action model before acquisition. Then freeze a source-fitted context-by-plan risk matrix and report its crossing witnesses, nominal design, robust design and least-favorable workload. Change only evaluation weights across predeclared query contexts; keep within-context queries, conditional utilities, observations and cost ledgers fixed. Compare nominal VOI, robust mixture, information gain, coverage, random and query-aware acquisition as an explicitly privileged reference. Test whether source-predicted crossings and regret reduction survive on held-out families, and whether calibrated error bars forecast failure. Reweighting the three seen families is development, not confirmation.

The most promising method extension is a transferable utility bridge with adequate short/long-query probe coverage, or cost-aware selection of action subsets relative to a strong fusion baseline. A positive toy workload result cannot rescue a source model whose reachable policies already trail simple baselines. Main-track novelty would need a new, validated way of learning that bridge or demonstrably useful diagnosis across independent pipelines; these supporting identities alone are insufficient.

Run `python papers/learning-to-explore-search-environments/full_paper/theory/verify_workload.py` and `python -m unittest discover -s papers/learning-to-explore-search-environments/full_paper/theory -p 'test_*.py' -v`. Ten tests include independent subset-game and message-partition enumeration, Bayes-risk monotonicity, exact primal/dual certificates, and the workload-adversary counterexample.
