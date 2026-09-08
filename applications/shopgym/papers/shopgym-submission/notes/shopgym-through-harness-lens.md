# ShopGym (your paper) — review through the harness-optimization lens

- **Paper:** arXiv:2605.16116v1 (May 15, 2026). Savadikar, Zhao, Zhu (you), Li, Xie, Castelo, Wu, Wang — NCSU + Shopify.
- No public code repo found as of June 2026; not yet indexed by major search engines (very recent). The paper itself is the artifact here.

## What it is

Two coupled components solving the realism-vs-control tradeoff in e-commerce agent evaluation:

- **ShopArena**: live seed storefronts → anonymized human-readable spec M (design manual + structured attributes + statistics) → staged code generation of a self-contained sandbox shop. Exploration and generation are fully decoupled through M; M is an editable control surface. Multi-seed composition lets one sandbox span diversity no single storefront covers.
- **ShopGuru**: grounded task generation over the sandbox — deterministic generators for short-horizon primitives (search-exact/substitute, browse, filter, shipping, returns) + LLM-authored long-horizon journeys with a validator-driven polish loop (≤2 rounds, halts on residual failures rather than shipping flawed tasks).

Validation: 224 tasks, six shops (3 synthetic, 3 twins). Structural (AXTree depth, interactive-element counts, state-transition graphs) and behavioral (GPT-5-mini / GPT-5 / Gemini 3 Flash on real vs twin shops — success rates track). Long-horizon tasks: 47.9% (GPT-5-mini) to 62.5% (GPT-5) — meaningful headroom.

## Already-present harness-optimization DNA

The paper *already practices* what HL / Meta-Harness preach, in the generation pipeline:

- Execution–verification loop with fresh agent per iteration (Ralph technique), feedback through the filesystem — the same context-externalization argument Meta-Harness makes (state/implementation/evaluation contexts split).
- Rule-based verifiers + multimodal verification agent = layered feedback channels.
- Validator-as-critic polish loop = reflective mutation on flagged items only — a mini-GEPA.

What it does **not** yet do: optimize the *evaluation-side* harness. Two harnesses are used as-is (Appendix B: BrowserGym/AXTree, internal screenshot+AXTree), treated as fixed instruments to validate environments.

## The opportunity (the gap the four works fill)

ShopGym builds the gym; nobody is yet *training in it*. The paper's own assets make it the ideal substrate for harness search:

1. Resettable, stable environments → eliminates the non-stationarity that makes live-web harness search unreproducible (the paper's own argument, turned inward).
2. Binary verifier V per task → the frozen metric (Karpathy's `prepare.py` role).
3. Skill catalog (7 categories) → natural Pareto dimensions (GEPA) and recurring-error-pattern buckets (Meta-Harness).
4. Cheap task regeneration → fresh held-out sets on demand; the cleanest known defense against harness overfitting (GEPA's documented failure mode) and judge-gaming (autoresearch's Goodhart lesson).
5. Two existing harnesses + three models → ready-made baselines and a cross-model transfer test (Meta-Harness's strongest claim).

A "harness track" — given frozen ShopGym envs + tasks, search over the CUA harness — would also strengthen the paper's positioning from "evaluation framework" to "training and optimization framework," matching the RL-ready "Gym" branding.

## Notes on coverage

Fetched HTML was truncated mid-Appendix A: main body, results, conclusion, and references fully read; Appendices B–L (harness details, limitations, prompts) only seen via TOC and in-text citations. Worth re-checking Appendix B details directly in the source when drafting the harness-optimization extension.
