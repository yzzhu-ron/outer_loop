# Heuristic Learning (HL) — "Learning Beyond Gradients"

- **Author:** Jiayi Weng (Trinkle23897), OpenAI post-training; EnvPool/Tianshou author. Blog post, May 2026.
- **Blog:** https://trinkle23897.github.io/learning-beyond-gradients/
- **Repo:** https://github.com/Trinkle23897/learning-beyond-gradients (cloned in `../1-heuristic-learning/learning-beyond-gradients/`)

## Core idea

A coding agent (Codex, gpt-5.4) iteratively maintains a *Heuristic System (HS)*: programmatic policy + state detectors + tests + replays + experiment logs + memory. Learning happens through direct code edits driven by feedback (env reward, test failures, logs, videos) — no backprop, no weight updates. HL is the update process; HS is the maintained object.

Key claim: heuristics were never useless, just too expensive for humans to maintain. Coding agents change the *maintenance curve* — the same way spinning machines changed thread production.

## Key results (all reproducible from repo)

- Breakout: 387 → 507 → 839 → 864 (theoretical max), each jump from a diagnosed failure mode (stuck-loop breaker, fast-low-ball lead, late-game offset release).
- MuJoCo Ant: CPG gait 2291 → +yaw feedback → +harmonics → residual MPC → 6146. HalfCheetah 11836 (5-ep mean). Both in Deep RL range.
- VizDoom D3 Battle: pure cv2/NumPy screen CV, mean=557 over 10 seeds.
- Atari57: 342 unattended runs (57 games × 2 obs modes × 3 repeats). Median HNS ~0.32 @ 1M steps (native_obs), far above PPO at the same step count; 0.81 @ 9.7M. Best-mode-per-game median HNS 0.83 vs PPO2 0.80, CleanRL PPO 0.98.
- Montezuma: boundary case — 400 pts but as an open-loop 86-macro-action route; shows plain reactive `if/else` policies need richer program forms (macro-actions, recoverable search state, long-term memory).

## Concepts worth stealing

1. **HS anatomy**: policy, state representation, feedback channels, experiment records (trials.jsonl / summary.csv), replays/tests, memory, update mechanism. A single rule file is not an HS.
2. **Forgetting becomes an engineering problem**: old capabilities pinned as regression tests, fixed-seed replays, golden traces, written-down failed directions.
3. **Two mandatory operations**: absorb feedback AND compress history (an HS that only grows becomes a big ball of mud).
4. **Coupling complexity**: the level of interdependence (states, rules, tests, signals) an agent can maintain. Bounded by module boundaries, test coverage, observability on the code side; model capability, context, memory, tools on the agent side. Clear feedback raises the maintainable complexity ceiling.
5. **Task shape matters**: "write me a policy.py" produced mediocre results; "maintain a complete loop with probes → detectors → policy → trials → videos → failure inspection → simplify + regression" is what worked.
6. **Fair accounting**: env steps for probing/debug all count toward the budget; sample-efficiency comparisons are about env interaction, not total compute (coding-agent tokens are not counted — acknowledged openly).

## Impact / community feedback

- Picked up widely; covered by 36kr ("OpenAI post-training engineer proposes new paradigm hypothesis") and circulated on X (e.g., China Research Collective). Weng's credibility (ChatGPT initial release, GPT-4/4o/o-series/GPT-5 infra) drives attention.
- Community spin-offs already exist, e.g. `xisen-w/hl-imagenet` (testing the blog's own stated boundary: HL can't solve ImageNet with pure Python — perception needs NNs).
- Positioning: a candidate "next paradigm" after pretraining → RLHF → RLVR: "anything that can be continuously iterated on becomes solvable." Proposed hybrid: HL as fast System-1 data processor; NNs for perception; LLM periodically retrained from HL-curated data.

## Relevance to ShopGym

- The HS framing maps directly onto a CUA harness: state detectors = DOM/AXTree/screenshot parsers; policy = action-selection scaffolding; replays = trajectory logs; regression tests = ShopGuru task suites with binary verifiers.
- ShopGym's stable, resettable sandbox shops are exactly the "clear feedback, reproducible state" environment HL needs — live storefronts are not.
- The Atari57 batch protocol (same prompt template, N seeds, required output files trials.jsonl/summary.csv/README) is a ready-made template for batch harness-search experiments over ShopGuru tasks.
