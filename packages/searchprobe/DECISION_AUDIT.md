# Audit a finite source decision model

`searchprobe decision-audit` computes the remaining action uncertainty in a supplied source utility matrix. It returns an exact minimax regret radius, a randomized action mixture attaining that radius, and a sparse source-world prior certifying that a smaller radius is impossible **inside that finite model**.

This is a separate interface from the label-free `searchprobe audit` command. It requires utilities from labeled source tasks or an explicitly synthetic construction, source provenance, and declared assumptions. It does not infer utilities from query logs, update source-world beliefs from target observations, or provide a target-regret guarantee.

```sh
searchprobe decision-audit packages/searchprobe/examples/source_model_synthetic.json \
  --output /tmp/searchprobe-decision-report.json
```

Without installing the package, prefix the command with `PYTHONPATH=packages/searchprobe/src python3 -m`.

The bundled model is **entirely synthetic**. Its utility rows are `(0,1,1)`, `(1,0,1)`, and `(1,1,0)`. Every pair of worlds shares an optimal action, but all three worlds together do not. The exact radius is `1/3`, attained by a uniform action mixture and certified by a uniform world prior. Pairwise agreement alone misses this conflict. This demonstrates the calculation; it is not evidence about a real search service.

## Supply one explicit model

The input is one JSON object, not JSONL:

```json
{
  "schema_version": 1,
  "kind": "source_utility_model",
  "model_id": "synthetic-binary-demo",
  "actions": ["router-a", "router-b"],
  "world_ids": ["world-one", "world-two"],
  "utilities": [[1, 0], [0, 1]],
  "provenance": {
    "utility_source": "Entirely synthetic example; no measured labels",
    "source_split": "Mathematical construction only",
    "task_distribution": "One abstract decision per world",
    "synthetic": true
  },
  "assumptions": [
    "Both routers are available in both worlds; these worlds are the complete model for this example."
  ]
}
```

All displayed fields are required. Action names and world IDs must be unique nonempty strings. `utilities` has one row per world and one column per action, with values in `[0,1]`. Integers, finite decimal numbers, and rational strings such as `"1/3"` are accepted. The Python API also accepts `fractions.Fraction`. Unknown fields, empty world sets, inconsistent dimensions, and missing provenance are rejected.

For measured data, document the labeled source collection in `utility_source`, the actual training/validation split in `source_split`, and the task weighting used to estimate every utility in `task_distribution`; set `synthetic` to `false`. Actions should denote the same available router or policy in every world. A router may map query context to a search action. The oracle is the best supplied router in each world, which is not automatically a per-query oracle.

The metadata are caller declarations. The tool cannot verify label provenance, estimate sampling uncertainty from means alone, or establish that the source worlds cover a target environment. To analyze a subset of worlds, supply a separately documented model containing that subset; the tool performs no automatic exclusion based on noisy target observations.

## Interpret the certificate

For supplied utility matrix `U`, define the loss of action `a` in world `w` as `L[w,a] = max_a U[w,a] - U[w,a]`. The decision radius is:

`R = min_action_mixture max_world expected_loss`.

The dual chooses a world prior maximizing the loss of its best action. Primal and dual agreement certify the computed radius. There exists a dual optimum supported on at most the number of actions; other optimal priors may be more diffuse.

The report retains the utility matrix and includes:

- `decision_radius`: exact rational value plus a floating-point display value.
- `minimax_action_mixture`: the conditional model solution.
- `worst_case_source_prior`: a sparse adversarial certificate, not a learned posterior or observed target frequency.
- `certificate`: primal and dual values, exact duality gap, support size, and world/action losses for checking the result.
- `pairwise_sum_incompatibility`, `pairwise_radius_lower_bound`, and `common_optimal_actions`.
- Provenance, caller assumptions, required interpretation, and inference limits. Every report sets `target_regret_guarantee` to `false`.

A zero radius means the supplied worlds share an optimal action. It cannot establish that probe selection is exhausted, unseen worlds share that action, source estimates are accurate, or adaptation will work. An empty model is rejected instead of being reported as zero uncertainty.

## Python API and runtime limits

```python
import json
from pathlib import Path
from searchprobe.decisions import audit_decision_model, audit_decision_json

model = json.loads(Path("source-model.json").read_text())
report = audit_decision_model(model)
assert report["certificate"]["duality_gap"]["exact"] == "0"
# audit_decision_json("source-model.json") reads and validates the file directly.
```

The solver uses exact rational vertex enumeration, with no dependencies. It checks primal and dual solutions independently. Runtime grows combinatorially: a model with `n` worlds and `m` actions enumerates `2 * (binomial(n+m,n) - 1)` candidate linear systems. The default limit is 20,000; the CLI exposes `--max-vertex-systems`, and both APIs accept `max_vertex_systems=`. Raising the limit can be expensive, especially for high-precision rationals. Larger models should use a production LP solver instead of dropping source worlds merely to obtain a favorable radius.

Valid computations exit `0`; invalid models, exceeded work limits, input/output errors, and certificate failures exit `2`. JSON goes to `--output` or stdout; the short summary and its source-model scope go to stderr. The `audit` command and its output schema retain their existing behavior.

The solver is adapted from this repository's [finite decision experiments](../../papers/learning-to-explore-search-environments/theory/finite_decisions.py). The method is standard finite minimax statistical decision theory and LP duality. The [theory note](../../papers/learning-to-explore-search-environments/theory/theory.md) explains the assumptions, multiway counterexample, and distinction between source-model uncertainty and transferable probe information. No novelty is claimed for the mathematics or solver.
