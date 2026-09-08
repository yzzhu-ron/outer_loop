# Harness Optimization Review

> **Start with [`index.html`](index.html)** — interactive review of the four core works (tabs, diagrams).
>
> **Then [`lit-review.html`](lit-review.html)** — "The Outer Loop": publication-grade, blog-style literature review of the field (essay, work map in four threads, timeline, reading list).

Scope: works on **learning beyond gradients**—improving AI systems by editing prompts, code, memory, and harnesses with LLM proposers while model weights remain frozen. The corresponding ShopGym application record is in [`../../applications/shopgym/`](../../applications/shopgym/).

## Contents

| Folder | Work | Artifact |
|---|---|---|
| `1-heuristic-learning/` | Heuristic Learning (HL), Jiayi Weng, May 2026 | Full blog repo clone (code, trials, CSVs, videos) |
| `2-gepa/` | GEPA: Reflective Prompt Evolution, Agrawal et al., ICLR 2026 | Repo clone (`gepa-ai/gepa`) |
| `3-karpathy-auto-research/` | autoresearch, Andrej Karpathy, Mar 2026 | Repo clone (`karpathy/autoresearch`) |
| `4-meta-harness/` | Meta-Harness, Yoonho Lee et al. (Stanford), Mar 2026 | Repo clone (`stanford-iris-lab/meta-harness`) |
| `notes/` | Per-work review notes (plain-text sources for index.html) | Markdown |

## One-paragraph synthesis

All four works run the same outer loop — propose a change to a text/code artifact, evaluate against a measurable metric, keep or discard, accumulate lessons — but differ in what is mutated and how feedback is consumed. GEPA mutates *prompts/text* with LLM reflection over execution traces and Pareto-aware selection. Karpathy's autoresearch mutates *training code* with a single-file hill-climbing ratchet on one scalar metric. Heuristic Learning mutates a whole *software system* (policy + detectors + tests + replays + memory) and frames maintenance/compression as the continual-learning mechanism. Meta-Harness mutates the *harness itself* (retrieval, memory, context assembly) with a coding-agent proposer that has filesystem access to all prior candidates' code, scores, and raw traces.
