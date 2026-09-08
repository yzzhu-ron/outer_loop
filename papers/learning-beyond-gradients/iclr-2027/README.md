# Resources, Certificates, and Limits of LLM-Guided Artifact Search (v2.5, ICLR 2027)

Major revision of [`../icml-2026/`](../icml-2026/) addressing both referee reports in full
([review](../icml-2026/review-icml2026.html), [verification report](../icml-2026/referee-report-r2-verification.html)).
The point-by-point mapping from every finding (W1–W9, Q1–Q6, E1–E4, N1–N4, R1–R11) to its fix
is in `response-to-reviewers.html` (open in a browser).

## v2.5 (minor revision over v2)

- **Retitled**: dropped the "Learning Beyond Gradients:" prefix so the paper no longer
  duplicates the title of the published blog it cites (Weng 2026). The phrase survives
  in lowercase as the name of the phenomenon.
- **ShopGym decoupled**: all citations to the ShopGym paper removed (unrelated work);
  the web-agent scaling discussion now cites WebArena only.
- **Completeness over compression**: the page limit no longer trades against clarity —
  new **Appendix M** (extended discussion: each falsifiable prediction P1–P5 paired with
  its measurement design, expanded limitations, outlook) and **Appendix N** (notation
  table) restore and exceed the prose trimmed for the 9-page body. Main text unchanged
  in substance and still ends on p. 9.
- `main-longversion.pdf` (untracked) is the pre-trim reference build; safe to delete
  once v2.5 is reviewed.

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Vendored: `iclr2027_conference.sty` (official ICLR style, year-stamped; includes
`\iclrarxivcopy` option) and `iclr2027_conference.bst` (official ICLR bst, provenance
header inside). Swap in the official kit when `github.com/ICLR/Master-Template`
publishes `iclr2027.zip` — historically a year-stamp swap. Compiles clean with
TeX Live; output `main.pdf` (20 pp total).

Figures regenerate via `cd code && python3 simulations.py` (NumPy + Matplotlib,
seeded, <1 min). The script prints every numeric claim made in §5/§9 for audit.

## Submission-requirements check (ICLR 2027, per the ICLR 2026 Author Guide)

- Main text **≤ 9 pages** at submission (10 at camera-ready) → body ends exactly on p. 9 ✓
- Ethics statement + reproducibility statement after main text, uncounted ✓ (p. 10)
- References unlimited ✓ (pp. 10–14); appendices after bibliography ✓ (pp. 15–20)
- Double-blind, anonymous, line-number ruler (review mode default) ✓
  Camera-ready: uncomment `\iclrfinalcopy`; arXiv: uncomment `\iclrarxivcopy`
- LLM-usage disclosure (required since ICLR 2026 when LLMs play a significant role) → Appendix L ✓
- Cycle: ICLR 2027 abstracts ~Sept 19, full papers ~Sept 24, 2026 (AoE), per historical pattern;
  confirm at iclr.cc when the CFP posts.

## What changed from v1 (summary; full map in response-to-reviewers.html)

**Score-moving additions (the reviewers' R9/R10, Q1/Q2):**
- **Theorem 6.1 (certified improvement at the compound rate)** — the composed theorem both
  reviewers asked for: prior mass × noise–gap × gated selection in one statement, false-reject
  slowdown explicit as (1−δ/B). Proof in Appendix E.
- **Proposition 3.5 (the boost is estimable)** — γ is identifiable to relative error η from
  K = O(ln(1/δ)/(η²q²_min)) verified proposals; pre-registered measurement protocol in
  Appendix I (120 buggy programs, 70 patches/condition, Wilson intervals, falsifier stated).

**Mandatory fixes (W3–W6 / E1–E4 / N1–N4):**
- Thm 5.4: asymmetric ascent f(i)=2(i−k) → unique optimum; boundary convention in statement.
- Prop 5.2: converse corrected to ln(1/(4δ)); new one-sided part (iii) (constant 8) so the
  N=85 gate is cited honestly via Cor 5.3(ii).
- Prop 8.1(iii) (was 7.1(iii)): restated inside F:A→[0,1]; range-filling construction.
- Prose numbers: 3.39σ median (3.90σ is the ceiling), ≈8.2 accepts (H₂₀₀₀=8.18),
  2 gated false accepts vs Poisson prediction 1.61 — the old "zero" kept visible as an
  audit note (§9, App. H), per R2's suggestion.
- Thm 5.1: empty-incumbent convention in statement; H_{t+1}−1 variant noted.
- Remark 3.4: exact-mass version of the γ<1 claim; §3 prose no longer overreaches Prop 3.3.
- Figures: all three feedback arms now execute real algorithms (assertion-checked);
  Fig 2b log-scale inset shows the gated rate honestly; captions distinguish executed
  algorithms from exact laws.

**Framing / sources (W1, W8, Q6, R11):**
- Title and abstract rewritten; "unifying theory" dropped in favor of what is proven.
- Selection results re-anchored to archival literature: Henderson et al. 2018,
  Dodge et al. 2019, Agarwal et al. 2021 (new "Evaluation statistics" paragraph in §10).
- Informal sources (autoresearch repo, HL blog) dagger-flagged in Table 1, demoted to
  illustrations; GEPA venue updated to ICLR 2026 (oral).

## Verified numbers (seed 0, byte-stable vs both referee replications)

| Claim | Value | Where |
|---|---|---|
| Ratchet median best, t=2000 | 3.393σ (mean 3.440σ; ceiling 3.899σ) | §5.1, Fig 2a, App H |
| Single-eval accepts | 8.30 simulated vs H₂₀₀₀ = 8.18 | §5.1, Fig 2b |
| Gate | N = ⌈84.77⌉ = 85; p_N = 2.02×10⁻⁶ | Cor 5.3(ii), App H |
| Gated false accepts | 2 / 8×10⁵ (Poisson prediction 1.61) | Fig 2b inset, App H |
| Tolerant-greedy exact/bound ratio at k=25 | 4.08 | App H |
| Archive means k=5/10/25 | 111 / 420 / 2551 (bounds 220/840/5100) | App H |

## Paper structure

| § | Content | Results |
|---|---------|---------|
| 1 | Introduction: seven contributions | — |
| 2 | Formal model + systems table + loop figure | Def 2.1 |
| 3 | Prior mass; misleading feedback; γ estimable | Thm 3.2, Prop 3.3, Rem 3.4, Prop 3.5 |
| 4 | Feedback information | Thm 4.1, Thm 4.2 |
| 5 | Selection: ratchet illusion + sound gating; deception | Thm 5.1, Prop 5.2, Cor 5.3, Thm 5.4 |
| 6 | **Composition: certified improvement** (new) | **Thm 6.1** |
| 7 | Occam bound; U-shape | Thm 7.1, Cor 7.2 |
| 8 | Goodhart limits (in-model) | Prop 8.1 |
| 9 | Simulations (all arms executed) + audit note | Figs 2–4 |
| 10 | Related work (incl. evaluation statistics) | — |
| 11 | Discussion: dials, limitations, conclusion | — |
| A–G | Proofs (E = composed theorem) | — |
| H | Experimental details + audit trail | — |
| I | **Pre-registered γ-measurement protocol** (new) | — |
| J | Open problem: self-reference rates | — |
| K | Extended system mapping | — |
| L | LLM usage disclosure (ICLR policy) | — |
| M | **Extended discussion: predictions P1–P5 with measurement designs, limitations, outlook** (new in v2.5) | — |
| N | **Notation table** (new in v2.5) | — |

## Next steps (stated in the paper as pending)

1. Run the Appendix I γ measurement (~1 GPU-day with a small open model).
2. The (ℓ(a), n, val–test gap) curve for Cor 7.2 from a deployed prompt optimizer.
3. Scale the γ protocol to web-agent harness search — a separate project, kept
   uncited in this paper; see the [ShopGym harness-track proposal](../../../applications/shopgym/papers/shopgym-submission/notes/harness-track-proposal.md).
