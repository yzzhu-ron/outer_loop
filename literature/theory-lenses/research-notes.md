# Research notes — theoretical foundations of learning beyond gradients

Working notes behind [`one-loop-many-names.html`](../../papers/learning-beyond-gradients/companion/one-loop-many-names.html). Each lens: key works, claims, and the mapping to the
empirical systems (HL, GEPA, autoresearch, Meta-Harness, SkillOpt, ADAS, DGM, AlphaEvolve…).

## The formal skeleton

A learning-beyond-gradients system is a tuple (A, f, q, S, M):
- A — artifact space (prompts, skill docs, harness code, policy programs)
- f — evaluator, noisy, expensive: f(a) = score + ε; sometimes also emits trace τ(a)
- q — proposer: q(a' | a, τ, M) — a frozen LLM conditioned on artifact, traces, and accumulated memory
- S — selection operator (greedy ratchet / Pareto archive / lineage / regression-gated)
- M — accumulated history (trials, replays, failure notes)

Every system in the empirical review is a point in this design space. Learning = improving the artifact;
the LLM weights never move. The theory question: what governs convergence rate, generalization of a*, and
failure (Goodhart, collapse)?

## Lens 1 — Universal search with learned priors

- Levin (1973), universal search: enumerate programs p in order of 2^{-l(p)}·t budget; total time
  ≤ c · t(p*)/P(p*) for the best verifier-passing program p*. **Search cost is inverse prior mass.**
  Scholarpedia: http://www.scholarpedia.org/article/Universal_search
- Solomonoff induction — universal prior over computable hypotheses; commonly called the "spiritual
  ancestor" of LLM next-token prediction.
- Schmidhuber: Adaptive Levin Search (Wiering & Schmidhuber 1996) — update proposal probabilities after
  each solved problem (≈ what trace-conditioning does per-iteration); OOPS (2004) — bias-optimal
  incremental program search reusing earlier solutions (≈ skill libraries / candidate archives);
  Gödel machine (2003) — the self-referential limit case. https://arxiv.org/abs/cs/0309048
- Modern bridges (verified June 2026):
  - "AI Agents as Universal Task Solvers: It's All About Time" — arXiv:2510.12066 (Amazon AGI / Soatto
    et al.): generalizes Levin/Solomonoff results to LLM-powered agents; intelligence as search-time
    speedup from learned priors. https://arxiv.org/abs/2510.12066
  - "LLM Priors for ERM over Programs" — arXiv:2510.14331: empirical risk minimization over program
    space with LLM as the prior. https://arxiv.org/abs/2510.14331
  - "From Levin's Universal Search to Policy-Guided Tree Search" — Entropy 28(4):434, 2026:
    https://doi.org/10.3390/e28040434 (policy-guided search inherits Levin-style guarantees)
- Mapping: GEPA/Meta-Harness/HL sample efficiency = high prior mass of frontier LLMs on "useful edits."
  AlphaEvolve/FunSearch = population-based Levin search with learned proposers. Voyager/OOPS parallel:
  solution reuse changes the effective prior for later problems.

## Lens 2 — Feedback as information

- "Provably Learning from Language Feedback" — arXiv:2506.10341 (2025): formal LLF setting; **transfer
  eluder dimension** quantifies information in feedback; theorem-level separation — reward-only can need
  exp(L) interactions where language feedback needs O(1)-dimension; "where the first mistake is" feedback
  enables stage-wise decomposition (exponential speedup); corrective feedback removes action-space
  dependence. https://arxiv.org/abs/2506.10341
- LLF-Bench (MSR, arXiv:2312.06853) — benchmark for interactive learning from language feedback (same
  group as Trace / Ching-An Cheng).
- "Language Models Can Learn from Verbal Feedback Without Scalar Rewards" — arXiv:2509.22638: scalar
  collapse loses information (two different critiques → same 0.8).
- "Expanding the Capabilities of Reinforcement Learning via Text Feedback" — arXiv:2602.02482.
- Mapping: this is the formal version of the field's most-rediscovered finding (GEPA's traces vs GRPO's
  scalars, 35x fewer rollouts; Meta-Harness's raw histories; TextGrad's textual gradients; HL's
  videos/logs). Blog framing: feedback channel capacity sets the convergence rate.

## Lens 3 — Selection under noise & diversity

- Best-arm identification / racing: Hoeffding races (Maron & Moore 1994); successive halving & Hyperband
  (Li et al. 2017). Repeats needed scale ~ σ²/Δ² — why autoresearch's noisy 5-min runs produced viral
  false positives (the unmerged "53% speedup"), and why ShopGym-style N-repeat aggregation is principled.
- Quality-diversity: Novelty Search (Lehman & Stanley 2011, "abandoning objectives"); MAP-Elites
  (Mouret & Clune 2015); book-length argument "Why Greatness Cannot Be Planned" (Stanley & Lehman 2015).
  Deceptive objectives ⇒ keep stepping stones. GEPA's instance-Pareto frontier, ADAS's archive, DGM's
  expanding lineage are all QD machinery in text space. Note Clune connects QD → ADAS → DGM personally.
- Empirical-theory bridge (2026): "Auto Researching, not hyperparameter tuning: Convergence Analysis of
  10,000 Experiments" — arXiv:2603.15916: autoresearch-style loops follow power-law convergence
  (c≈0.11); frames LLM agents as contextual search policies with information-theoretic diagnostics
  (entropy, JSD, innovation rate). https://arxiv.org/abs/2603.15916

## Lens 4 — The artifact as compression

- MDL (Rissanen 1978) / Kolmogorov complexity / Hutter's AIXI (2005): learning = compression.
- The evolved artifact (prompt, playbook, skill doc, harness) is a *compressed sufficient statistic of
  the task distribution*, externalized in text. PAC-Bayes over a discrete artifact space: generalization
  gap controlled by description length / prior mass of a* — overly long, overfit artifacts generalize
  worse. (GEPA's observed test-degradation; SkillOpt's bounded-edit + held-out acceptance discipline is
  exactly description-length control.)
- HL: "absorb feedback AND compress history" — compression as explicit regularization. ACE's "context
  collapse" and "brevity bias" = the two failure directions of description-length management
  (over-compress vs never-compress).
- Inner mechanism — ICL theory: ICL as implicit Bayesian inference (Xie et al. 2021, arXiv:2111.02080);
  transformers doing in-context gradient descent (von Oswald et al. 2023, arXiv:2212.07677); Garg et al.
  2022 (arXiv:2208.01066). The artifact is the posterior/dataset the frozen model conditions on —
  "learning without weight updates" is then literal, just relocated into context.

## Lens 5 — Limits & failure theory

- No Free Lunch (Wolpert & Macready 1997): averaged over all problems, no search beats random — so all
  observed gains come from prior–problem alignment. Makes the "LLM = learned prior" claim load-bearing,
  and predicts domain-dependence (HL's Montezuma failure; GEPA's weak-signal domains).
- Goodhart taxonomy (Manheim & Garrabrant 2018, arXiv:1803.04585): regressional / extremal / causal /
  adversarial — outer loops hit extremal+adversarial (DGM hallucinating tool logs; Eureka reward
  hacking; autoresearch metric gaming). Proxy gap grows with optimization pressure.
- RSI limits: Yampolskiy ("From Seed AI…", limits of recursive self-improvement); Chalmers 2010
  singularity analysis — fixed-point/divergence framing for improve(improver) (STOP's question).
- Counterpoint to lens 2: "Reward is Enough" (Silver, Singh, Precup, Sutton 2021) — scalar reward
  suffices *in the limit*; the beyond-gradients corpus is effectively an empirical argument that it is
  not enough *at practical sample budgets*.

## Conjectures for the essay (the paper-idea payload)

C1 (runtime): trace-conditioned proposers inherit adaptive-Levin-search bounds — expected hitting time of
an ε-good artifact ≤ c · Σ t(aᵢ)/q(path to a*), with q boosted by pretraining alignment. Testable: scaling
of hitting time with proposer quality.
C2 (channel capacity): convergence rate upper-bounded by mutual information I(feedback; improvement
direction) per rollout; scalar reward ≈ low-capacity channel; trace ≈ high-capacity. Predicts the
GEPA/GRPO 35x gap class. (Builds directly on transfer eluder dimension.)
C3 (diversity): in deceptive text landscapes, archive/Pareto selection has polynomial hitting time where
greedy is exponential (transfer novelty-search results to text space).
C4 (noise): acceptance of an edit with true gain Δ under eval noise σ requires N ≳ σ²/Δ² repeats;
ratchets without repeats accumulate false positives at a predictable rate (the autoresearch pathology).
C5 (compression/generalization): PAC-Bayes bound for artifacts — test gap ≤ Õ(√(len(a*)/n)); predicts
SkillOpt's bounded-edits > unbounded rewriting at fixed budget, and ACE's collapse as bound violation.
C6 (self-reference): improve(improver) converges only if evaluator gap (true vs proxy objective) shrinks
faster than optimization pressure grows; otherwise Goodhart divergence — formal conditions open.

## Candidate venues / formats

High-quality blog first (Anthropic/OpenAI-blog register, the existing two-HTML house style), then
possibly a position paper ("Position: learning beyond gradients is universal search with learned
priors") — ICML position track or an ICLR/NeurIPS workshop on open-endedness / self-improving systems.
