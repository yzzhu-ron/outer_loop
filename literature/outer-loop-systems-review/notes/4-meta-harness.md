# Meta-Harness — End-to-End Optimization of Model Harnesses

- **Paper:** arXiv:2603.28052, Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn (Stanford IRIS). Preprint Mar 30, 2026. Site: https://yoonholee.com/meta-harness/
- **Repo:** https://github.com/stanford-iris-lab/meta-harness (cloned in `../4-meta-harness/meta-harness/`); optimized TB2 harness in separate artifact repo `meta-harness-tbench2-artifact`.

## Core idea

An agentic *outer loop* that searches over harness code — the code around a fixed base model deciding what to store, retrieve, and show. The proposer is a coding agent (Claude Code in the reference implementation) given **unrestricted filesystem access to all prior candidates**: their source code, scores, and *raw uncompressed execution traces*. It causally debugs past failures and writes new harness candidates (retrieval logic, memory management, prompt assembly).

Key diagnosis: prior text-optimization methods (GEPA included) mutate prompts and summarize feedback; harness engineering needs the proposer to read *whole execution histories* and edit *arbitrary code*. The conceptually simple fix — expose a filesystem of prior experience to a coding agent — is the contribution.

## Key results

- Online text classification: +7.7 points over a SOTA context-management system with 4x fewer context tokens.
- Retrieval-augmented math reasoning: a single discovered harness improves 200 IMO-level problems by +4.7 avg across five held-out models (harnesses transfer across models).
- Agentic coding: discovered harnesses surpass best hand-engineered baselines on TerminalBench-2 — reported #2 among Opus-4.6 agents, #1 among Haiku-4.5 agents.

## Repo notes (practical)

- `ONBOARDING.md` is a structured interview prompt for adapting Meta-Harness to a *new domain* → produces `domain_spec.md` (problem framing, harness interface, evaluation, baselines, offline/online experience, budget). Built-in warnings about eval leakage and test-set contamination.
- Stated fit criteria match ShopGym almost line by line: long-horizon multi-step task ✓, repeated episodes ✓, fixed base model ✓, measurable success metric ✓, recurring error patterns ✓, stable eval loop ✓.
- Two reference examples: text-classification memory search, TB2 scaffold evolution. Proposer-agnostic via `claude_wrapper.py`.

## Impact / community feedback

- Strongly positive reception: arXivIQ, Hugging Face community report, Softmax Data ("A New Harness in Town"), Menon Lab coverage; NerdHeadz framed it as "Stanford just proved it's the harness, not the model." Japanese paper-notes community picked it up too.
- The TB2 leaderboard placement is the headline evidence that machine-discovered harnesses beat hand-engineering on a competitive public benchmark.
- Repo is fresh ("not been tested beyond verifying that it runs") — early-stage code, no large fork ecosystem yet.

## Relevance to ShopGym

- This is the closest blueprint for the proposed work: replace "TerminalBench-2 scaffold" with "CUA web-browsing harness," replace TB2 tasks with ShopGuru tasks. The TB2 reference example (`reference_examples/terminal_bench_2/`) is a working scaffold-evolution harness to adapt.
- ShopGym fixes Meta-Harness's hardest prerequisite — a stable, resettable, non-contaminated eval loop — which live-web benchmarks can't provide.
- The cross-model transfer result suggests a harness discovered with one model (e.g., GPT-5-mini for cheap search) may transfer to stronger models — directly testable with ShopGym's three-model eval setup.
- Use `ONBOARDING.md` to draft `domain_spec.md` for the ShopGym CUA domain as the literal first implementation step.
