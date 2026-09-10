# Source-only intervention utility transfer

The completed study did not show an improvement: the centered bridge reached
0.428981 nDCG@10 versus 0.429495 for the source-fixed policy using the same
160 labeled questions per source family.
The paired difference was −0.000514, with a conditional 95% interval spanning
zero. The registered advancement gate failed. See [RESULTS.md](RESULTS.md) for
the findings and independent audit; all frozen artifacts remain unchanged.
Fixed160 already matched each target family's best fixed policy on this test set;
even a perfect lambda-specific global choice could gain only 0.0023075 macro nDCG,
below the registered .005 gate. This limits what the benchmark can establish.

The registered test fits one scalar probe-to-utility slope per serving policy.
Within-source centering estimates backend response while an equal-source residual
intercept supplies the target reference. The target policy sees eight selected
document probes from one hidden backend. It never sees another target backend or
target relevance labels. The intercept's transfer is an explicit hypothesis.

There are eleven policies under a cap of two wrapper searches: five single
actions and six pairs of the four one-search actions. The primary centered model
is compared with pooled slopes, source-fixed policies using 128 and 160 labeled
questions per source family, source utility given privileged lambda, and twenty
complete within-source utility/lambda shuffles. A fixed-strength centered model
remains a diagnostic.
The practical advancement gate must hold against both source-fixed baselines:
at least +.005 macro nDCG, positive effects in at least two families, and no family
loss exceeding .005. The full registration is in `protocol.v1.json`.

Eight valid indirect-question bundles cost 48 wrapper/96 underlying searches,
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

All writers reject existing outputs. The code and prediction freezes were
committed as `16f668d` and `b6b55ef`. To replay safely in a new copy without raw
caches, run from this directory in a Python environment with NumPy installed.
The reference versions are Python 3.14.0 and NumPy 2.5.3; no model downloads,
SciPy, or scikit-learn are needed for this replay.

```sh
python -m unittest discover -s . -p 'test_*.py'
python replay_portable.py
```

The replay helper keeps the regenerated lock's new timestamp, checks that every
other decision field is identical, and evaluates the unchanged copied original
lock against its hash-bound labels. Evaluation reproduces byte-identical results.
It never rewrites frozen labels to accommodate a new timestamp.

The original export phases verified the actual selected raw cache directory
against the frozen source hashes. The helper requires the eight implementation
files listed by the freeze plus five artifacts: `freeze.v1.json`, `inputs.v1.json`,
`decisions.v1.json`, `test_labels.v1.json`, and `results.v1.json`. These are portable
repository files; ignored raw caches are unnecessary. The inputs preserve each document's policy
RR vector and all five paid action ranklists, enabling compression and alias
audits without retrieval. The freeze includes the transitively imported keyword
engine.

Each target query's seven-lambda/three-panel/method vector stays together in
bootstrap resampling. Families and probe panels are fixed; panel variation is
reported separately. These are three previously explored development families,
and the mixtures reuse truncated endpoint rankings. Conditional intervals do not
establish population generalization or a new-family confirmation.
