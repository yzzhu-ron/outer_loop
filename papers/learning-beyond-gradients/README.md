# Learning Beyond Gradients

This paper project develops a theory of LLM-guided search over prompts, code,
memory, harnesses, and other discrete artifacts. Its central abstraction is a
proposal distribution informed by execution feedback, an expensive and noisy
evaluation, and a rule for retaining candidates.

## Manuscript generations

- [`iclr-2027/`](iclr-2027/) is the current manuscript, **Resources,
  Certificates, and Limits of LLM-Guided Artifact Search**. It adds a composed
  improvement theorem, an estimable trace-conditioning factor, corrected noisy
  selection results, a preregistered measurement protocol, and the full review
  response record.
- [`icml-2026/`](icml-2026/) is the earlier manuscript, **Learning Beyond
  Gradients: A Unifying Theory of LLM-Guided Artifact Search**, with its two
  detailed referee reports.

Each generation retains its own LaTeX style, source, compiled PDF, figures, and
seeded simulation code so that its historical claims remain reproducible.

## Companion material

- [`companion/one-loop-many-names.html`](companion/one-loop-many-names.html) is
  the interactive essay that first assembled the five theory lenses.
- [`companion/blog/`](companion/blog/) contains “The 53% Speedup That Wasn't,”
  its source Markdown, build script, and generated figures.
- [`../../literature/theory-lenses/research-notes.md`](../../literature/theory-lenses/research-notes.md)
  is the underlying research notebook.
- [`../../literature/outer-loop-systems-review/`](../../literature/outer-loop-systems-review/)
  compares the empirical systems that motivated the abstraction.

## Theory lenses

1. **Universal search and learned priors:** proposal efficiency depends on how
   much probability the proposer places on useful edits.
2. **Feedback as information:** language and structured traces can identify
   useful changes more efficiently than scalar reward alone.
3. **Selection under noise:** repeated evaluation and confidence-aware gates
   are part of the learning algorithm, not reporting details.
4. **Archives and deceptive landscapes:** keeping diverse partial solutions can
   avoid commitment to a locally attractive artifact.
5. **Description length and limits:** artifact complexity trades approximation
   error against validation overfitting, under the usual prior-alignment and
   Goodhart constraints.

The theory paper is a general track. Empirical manuscripts tied to one
application are stored with that application.
