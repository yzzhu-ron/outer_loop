# `outer_loop` Python package

This package contains the application-neutral control-plane pieces shared by
outer-loop research applications.

The public surface is small:

- `Proposer`, `Evaluator`, and `Evaluation` define the application boundary.
- `wilson_ci` and `gated_accept` implement the binary-outcome confidence gate.
- `GatedRatchet` manages a statistically gated incumbent.
- `ParetoArchive` retains non-dominated candidates and chooses parents with UCB1.

An application supplies the artifact representation, trace format, evaluator,
and proposer. See `applications/shopgym/` for the first concrete integration.
