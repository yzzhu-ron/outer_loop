# Learning Beyond Gradients: A Unifying Theory of LLM-Guided Artifact Search

Full ICML-format theory paper developed from the [research notes](../../../literature/theory-lenses/research-notes.md) and the essay
[`one-loop-many-names.html`](../companion/one-loop-many-names.html). Conjectures C1–C5 from the notes are now theorems with complete
proofs; C6 (self-reference rates) is stated as a formal open problem (Appendix H).

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Everything needed is vendored: `icml2026.sty`, `icml2026.bst`, `algorithm.sty`, `algorithmic.sty`
(from the official ICML 2026 kit). Compiles clean with TeX Live; output `main.pdf` (14 pp total).

Figures regenerate via `cd code && python3 simulations.py` (NumPy + Matplotlib, seeded, <1 min).

## Submission-requirements check (ICML 2026 CFP)

- Main body **must be ≤ 8 pages**; references, impact statement, and appendices unlimited.
  → Body ends on p. 7 ✓ (one page of slack for revisions). Refs pp. 7–10, appendix pp. 11–14.
- Anonymized double-blind + line-number ruler → using review mode (default) ✓.
  Camera-ready: switch to `\usepackage[accepted]{icml2026}` and fill the real author block (marked
  `TODO` comment in `main.tex`).
- Single PDF ≤ 50 MB ✓. LaTeX only ✓. Impact statement included (excluded from page count) ✓.
- Cycle note: the next open deadline is ICML 2027 (~late Jan 2027; style files not yet released —
  historically a year-stamp swap). ICLR 2027 (~Sept 2026) is the nearest alternative; the body
  fits ICLR/NeurIPS 9-page limits with room to spare.

## Paper structure

| § | Content | Results |
|---|---------|---------|
| 1 | Introduction: six contributions mapped to systems | — |
| 2 | Formal model: artifact-search system (A, F, q, sel, M); Table of 10+ systems; loop figure | Def 2.1 |
| 3 | Prior mass ⇒ hitting time (adaptive Levin bound) | Thm 3.2, Prop 3.3 |
| 4 | Feedback information: transcript bound; scalar-vs-trace exponential separation | Thm 4.1, Thm 4.2 |
| 5 | Selection: ratchet illusion σ√(2ln t), ln t false accepts; N≳σ²/Δ² repeats (matching lower bound); deception: greedy ∞ / ε-greedy exp(k) / archive O(k²) | Thm 5.1, Prop 5.2, Cor 5.3, Thm 5.4 |
| 6 | Occam bound for artifacts; U-shape: context collapse vs validation bloat | Thm 6.1, Cor 6.2 |
| 7 | Goodhart: bounded-bias transfer, optimizer's curse √(2ln m), evaluator-integrity necessity | Prop 7.1 |
| 8 | Simulations validating Thms 4.2 / 5.1 / 5.4 (figures from `code/`) | Figs 2–4 |
| 9–10 | Related work; discussion: five measurable dials, falsifiable scalings | — |
| A–F | Complete proofs | — |
| G | Experimental details | — |
| H | Open problem: rates for self-referential improvement (was C6) | — |
| I | Extended system mapping (GEPA, SkillOpt, ACE, Meta-Harness, autoresearch, HL, ADAS/DGM, FunSearch/AlphaEvolve, Voyager, ShopGym/CUA) | — |

## Theorem-to-evidence map (for rebuttals)

- Thm 5.1 ↔ autoresearch noisy-eval false positives (the unreplicated "53% speedup"); Fig. 2.
- Thm 4.2 ↔ GEPA-vs-GRPO 35× rollout gap; TextGrad/Trace/Reflexion trace advantage; Fig. 3.
- Thm 5.4 ↔ GEPA Pareto frontier, ADAS archive, DGM lineage; Fig. 4.
- Cor 6.2 ↔ ACE context collapse + GEPA val overfitting + SkillOpt bounded edits.
- Prop 7.1(iii) ↔ DGM hallucinated tool logs, Eureka reward hacking.

## Provenance / verification status

All cited arXiv IDs and author lists were verified against arXiv/web in June 2026, or taken from
the [verified reading list](../../../literature/theory-lenses/research-notes.md). Long author lists use "and others"
deliberately. ShopGym is cited as Anonymous (under review) for double-blind hygiene.
Two refs from the notes were dropped pending author verification: arXiv:2510.14331 (LLM priors
for ERM) and arXiv:2603.15916 (autoresearch convergence) — re-add if wanted.
