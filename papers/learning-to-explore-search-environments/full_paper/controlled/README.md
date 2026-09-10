# Controlled acquisition suite

This exact finite experiment separates task relevance, planning horizon, and model alignment. It supplies constructed source models; it does not fit a retrieval model or establish that these mechanisms improve natural search.

The XOR panel contains independent decision bits, a complementary pair, two nuisance bits, and a noisy direct proxy. The gated panel additionally makes the useful second probe depend on the first observation. All methods receive the same candidate channels and use the same posterior router. The fixed comparator exhaustively chooses its best nonadaptive probe subset under the source model; it is deliberately strong.

The [protocol](protocol.v1.json) and executable/test hashes were [frozen](freeze.v1.json) at **2026-09-10 13:46:30 UTC**, before the registered grid ran. Seven mechanical tests and independent analytical checks preceded the freeze. The run completed 1,658 method/budget rows across 38 conditions in 25.81 seconds. Expected utility, regret, and operations are exact rational numbers; entropy calculations use floating-point logarithms. No Monte Carlo confidence intervals apply.

Read [RESULTS.md](RESULTS.md) for findings and limitations. The principal outputs are:

- [table.json](results.v1/table.json): every registered method, weight, budget, and control; both fraction strings and numeric fields.
- [summary.json](results.v1/summary.json): provenance and the predefined illustrative cases.
- [model_diagnostics.json](results.v1/model_diagnostics.json): initial acquisition scores and two-way Blackwell incomparability witnesses for every probe pair.
- [condition_timings.json](results.v1/condition_timings.json): elapsed time and memoized state counts.

From this directory:

```sh
python3 -m unittest -v
python3 experiment.py --freeze
python3 experiment.py --run
```

An existing freeze is never replaced when hashes change; the run rejects code/protocol drift. A revised design must use a new version. Re-running unchanged code reproduces all exact values; wall-clock fields can vary. The standard library is sufficient.

For embedding, filter `table.json` by `control == "matched"`, `source_weight == "3/4"`, `budget == 2`, and `scenario`. Fields `expected_utility_exact` and `expected_utility` provide exact and numeric values. `context` identifies parity versus the gate-selected bit. Hidden/announced task-mix controls, five utility-row shuffles, noise controls with a channel-calibrated DP comparator, and candidate-order sensitivity are preserved in that same table.
