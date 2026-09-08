# Synthesis: Optimizing a CUA harness for web browsing with ShopGym

How the four works compose into one concrete plan.

## The works on one axis

| | Mutated object | Proposer | Feedback consumed | Selection |
|---|---|---|---|---|
| GEPA | Prompts / any text | Reflection LLM | Execution traces (summarized into reflection) | Pareto frontier over instances |
| autoresearch | One code file | Coding agent | Single scalar (val_bpb) | Greedy ratchet (keep/revert) |
| Heuristic Learning | Whole software system | Coding agent | Reward, tests, logs, videos, replays | Regression-gated (simplify + verify no regress) |
| Meta-Harness | Harness code | Coding agent + filesystem of ALL prior candidates/traces | Raw uncompressed traces | Score-ranked, agent decides |

Reading order of generality: autoresearch (simplest ratchet) ⊂ GEPA (population + reflection) ⊂ Meta-Harness (arbitrary code + full history) ≈ HL (adds the maintenance/compression discipline that keeps the searched artifact alive long-term).

## Proposed experiment: ShopGym harness track

**Frozen** (the `prepare.py` of the experiment): sandbox shops, ShopGuru task suites, rule-based verifiers + LLM judge, base model(s), step/token budgets.

**Mutable**: the CUA harness — observation builder (AXTree pruning, screenshot policy, history compression), action space definition, system prompt, retry/recovery logic, memory across steps, termination logic.

**Loop** (Meta-Harness structure, autoresearch discipline, GEPA selection, HL hygiene):

1. Seed with the two existing ShopGym harnesses (BrowserGym-based and internal) as baseline candidates.
2. Each iteration: a coding-agent proposer reads the candidate library — code, per-skill-category scores, raw trajectories of failures — and writes a new harness candidate.
3. Evaluate on a fixed search split of ShopGuru tasks, N repeats per task (eval noise is real; autoresearch's false-positive lesson).
4. Selection: keep a Pareto set over the 7 skill categories rather than one global best (long-horizon journeys and filter tasks likely favor different observation/action tradeoffs).
5. HL hygiene: every accepted candidate must pass regression replays of previously-solved tasks; periodic "compression" iterations where the only goal is simplifying the best harness without score loss.
6. Report on held-out: (a) unseen tasks on seen shops, (b) a fully unseen generated shop, (c) cross-model transfer (search with GPT-5-mini, test on GPT-5 / Gemini 3 Flash).

**Anti-Goodhart guards**: rule-based verifiers are ground truth where available; LLM-judge audited on a sample; regenerate fresh tasks for final numbers (ShopGym's unique capability — none of the four works could do this).

## Why this is a strong paper extension

- Meta-Harness showed harness search wins on TerminalBench-2 but had to fight for a clean eval loop; ShopGym *is* the clean eval loop for the CUA/web domain. The combination is novel: nobody has run agentic harness search on a fully regenerable web benchmark.
- It converts ShopGym's headroom result (47.9–62.5% on long-horizon) from a static observation into a research instrument: how much of the gap closes via harness alone, with the model frozen?
- Cross-model transfer of discovered harnesses (Meta-Harness's IMO result) is directly testable and, if it holds on web tasks, is a headline result on its own.

## Concrete first steps

1. Run Meta-Harness `ONBOARDING.md` (in `../4-meta-harness/meta-harness/`) against the ShopGym domain → `domain_spec.md`.
2. Adapt `reference_examples/terminal_bench_2/` scaffolding: swap TB2 runner for the ShopGym eval harness; keep `claude_wrapper.py` logging pattern.
3. Write the `program.md` (autoresearch-style) defining constraints, frozen surfaces, simplicity criterion.
4. Cheap pilot: GEPA over the harness *system prompt only* (lowest effort, uses `gepa.optimize_anything`) to get a baseline for "text-only" gains before unlocking full code mutation — this gives the paper an ablation for free: prompt-level vs code-level harness optimization.
