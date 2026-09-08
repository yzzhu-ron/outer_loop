# ShopGym — NeurIPS submission record

This folder holds the material used to understand the ShopGym paper, prepare
review responses, and formulate a follow-up harness-search study.

- Paper: https://arxiv.org/abs/2605.16116 (v1, May 15, 2026)
- `notes/shopgym-through-harness-lens.md` — review of the paper through the harness-optimization lens (strengths, gaps, the "evaluation harness is fixed" opening)
- `notes/harness-track-proposal.md` — proposed harness-search experiment (Meta-Harness structure + autoresearch discipline + GEPA selection + HL hygiene) — candidate camera-ready addition or follow-up paper
- `SOURCES.md` — paper links

## Related work to track for the rebuttal (CUA environments + benchmarks)

Direct competitors / must-know baselines, mostly from the paper's own reference list plus the 2026 generative wave:

| Work | What it is | Positioning vs ShopGym |
|---|---|---|
| WebShop (NeurIPS 2022) | Synthetic storefront, 1.18M products | Hand-built sandbox; no generation pipeline |
| WebArena / VisualWebArena (ICLR/ACL 2024) | Self-hosted functional sites, executable tasks | Fixed sites; not e-commerce-grounded, not regenerable |
| BrowserGym (TMLR 2025) | Standardized web-agent harness ecosystem | ShopGym's baseline harness A |
| Mind2Web (NeurIPS 2023) | Offline traces, 137 real sites | Offline; no executable env |
| DeepShop (2025) | Live-website shopping benchmark | The non-stationarity problem ShopGym solves |
| ShoppingBench (AAAI 2025/26) | 2.5M-product sandbox, API interaction | Manual build; API not browser |
| ShopSimulator (2026) | Chinese RL shopping env, multi-turn dialog | Dialog-centric |
| WebMall (2025) | Four simulated shops, multi-shop tasks | Manual build, fixed four shops |
| AgenticShop (2026) | Personalized product-curation benchmark | Task-type complement |
| SimGym (2026, Shopify) | Traffic-grounded agents for offline A/B testing | Sister work — persona/traffic side |
| gym-anything (CMU, 2026) | Any software → agent environment | General-software generative wave |
| WebArena-Infinity (2026) | Generated browser envs w/ verifiable tasks at scale | Closest generative competitor |
| WebGym (2026) | Scaled training envs for visual web agents | Training-focus competitor |
| WebForge (2026) | "Realism-reproducibility-scalability trilemma" | Competing trilemma framing — read closely |
| Safe recreated websites (Chae et al., 2026) | Web-agent learning via recreated sites | Safety-motivated recreation — overlapping pipeline |
| OPERA (2025) | Human shopping-session traces (obs/persona/rationale/action) | Human-grounding data for task realism claims |

Likely rebuttal themes: (1) vs generative-env competitors — ShopGym's differentiators are the editable human-readable spec, validated structural/behavioral alignment, and per-task binary verifiers; (2) judge reliability — GPT-5 judge + rule verifiers; (3) scale questions — 224 tasks / 6 shops vs regenerability argument; (4) anonymization/IP safeguards.
