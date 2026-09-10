# Source-only intervention utility transfer

Implementation checkpoint only. No real interior-mixture outcomes, model fits,
prediction locks, or target utility exports have been produced here. The original
hybrid adapter and its frozen artifacts are unchanged.

The registered test fits one scalar probe-to-utility slope per serving policy.
Within-source centering estimates backend response while an equal-source residual
intercept supplies the target reference. The target policy sees eight selected
document probes from one hidden backend. It never sees another target backend or
target relevance labels. The intercept's transfer is an explicit hypothesis.

There are eleven policies under a cap of two wrapper searches: five single
actions and six pairs of the four one-search actions. The primary centered model
is compared with pooled slopes, source-fixed policies using128 and160 source
labels, source utility given privileged lambda, and twenty complete within-source
utility/lambda shuffles. A fixed-strength centered model remains a diagnostic.
The practical advancement gate must hold against both source-fixed baselines:
at least+.005 macro nDCG, positive effects in at least two families, and no family
loss exceeding.005. The full registration is in `protocol.v1.json`.

Eight valid indirect-question bundles cost48 wrapper/96 underlying searches,
eight sample calls, and sixteen generation calls with their recorded token
charges. Invalid attempts are retained and charged according to actual available
groups. Eta-zero and fixed policies do no onboarding. Results report full cost
vectors, amortization, and the six-wrapper-search RRF-all quality reference.

The pipeline separates provider access from policy access. The provider parses
the original monolithic prepared file, which physically contains all qrels.
Its first export computes source train/calibration utilities and document-probe
observations, plus test IDs and serving costs. It emits no test utilities. The
decision process reads only that export and saves every target decision. A second
export computes new test utilities only after the decision lock's hashes match.
UTC timestamps record these stages. The evaluator then computes paired metrics,
centered response error, action-specific intercept errors, and profile-alias
diagnostics; those target-centered quantities never enter policy inference.

All writers reject existing outputs. Freeze once before computing real interior
outcomes; use new explicit output paths for later reanalysis. From this directory,
with the existing pilot Python environment:

```sh
../../pilot/.venv/bin/python -m unittest discover -s . -p 'test_experiment.py'
../../pilot/.venv/bin/python experiment.py freeze
../../pilot/.venv/bin/python export_evidence.py inputs
../../pilot/.venv/bin/python experiment.py decide
../../pilot/.venv/bin/python export_evidence.py test
../../pilot/.venv/bin/python experiment.py evaluate
```

The export phases verify the actual selected raw cache directory against the
frozen source hashes. Decision/evaluation replay needs only the tracked frozen
implementation, `inputs.v1.json`, `decisions.v1.json`, and `test_labels.v1.json`;
ignored raw caches are unnecessary. The inputs preserve each document's policy
RR vector and all five paid action ranklists, enabling compression and alias
audits without retrieval. The freeze includes the transitively imported keyword
engine.

Each target query's seven-lambda/three-panel/method vector stays together in
bootstrap resampling. Families and probe panels are fixed; panel variation is
reported separately. These are three previously explored development families,
and the mixtures reuse truncated endpoint rankings. Conditional intervals do not
establish population generalization or a new-family confirmation.
