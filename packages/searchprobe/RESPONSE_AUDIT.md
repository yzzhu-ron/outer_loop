# Conditional router responsiveness

`response-audit` checks whether bounded score corrections can change a router's decisions. A large base-score margin may prevent an environment profile from changing any selected action, regardless of how informative its probes are. This command makes that limitation measurable before collecting more probes.

```sh
searchprobe response-audit model.json --output report.json
```

The calculation is exact **conditional on the supplied bounds**. The tool does not verify those bounds, run a router, read task labels, or establish target regret or generalization. The interval comparison is elementary robust argmax analysis, not a new theorem.

## Input contract, version 1

Supply one UTF-8 JSON object:

```json
{
  "schema_version": 1,
  "kind": "score_box_model",
  "model_id": "synthetic-example",
  "actions": ["original", "rewrite"],
  "row_ids": ["wide-margin", "reachable-tie"],
  "base_scores": [["0.8", "0.3"], ["0.4", "0.5"]],
  "correction_bounds": ["0.05", "0.05"],
  "provenance": {
    "score_source": "Handwritten synthetic scores",
    "bound_source": "Assumed bounds for demonstration",
    "row_population": "Two synthetic rows",
    "synthetic": true
  },
  "assumptions": ["Each score changes by at most 0.05 in absolute value."]
}
```

`actions` and `row_ids` must be nonempty arrays of unique, nonempty strings. `base_scores` has one row per row ID and one score per action. `correction_bounds` is a shared vector of nonnegative absolute bounds, one per action, applying to every row. Empty inputs, unknown fields, duplicate JSON keys, booleans in numeric fields, and nonfinite numbers are rejected. All provenance fields and at least one nonempty assumption are required; `synthetic` must be an explicit boolean.

Scores and bounds accept finite JSON numbers or decimal/rational strings such as `"0.1"` and `"1/3"`. JSON decimal numbers retain their written value. Python also accepts `Decimal` and `Fraction`; Python floats are interpreted through their decimal string. Arithmetic uses exact rational values. Reported numeric objects contain `exact` and an approximate `value`; `value` is `null` when conversion would overflow a finite float.

## Meaning of the certificate

For every row and action `j`, the declared model is:

```text
corrected_score[j] = base_score[j] + delta[j]
-bound[j] <= delta[j] <= bound[j]
```

The highest score wins, with the **lowest action index winning exact ties**. Input action order therefore matters. A row is certified unchanged exactly when the original winner's lower score still wins against every challenger's upper score, including that tie rule.

For example, base scores `[0.5, 0.4]` with bounds `[0.05, 0.05]` are certified unchanged: the worst case is a tie that action 0 wins. Reversing those scores makes a change possible because action 0 can displace the initial winner at the tie.

`potential_challengers` lists alternative actions that can actually win somewhere in the full independent score box. A candidate's upper score must beat every other action's lower score. Merely threatening the original winner is insufficient if a third action always blocks the candidate.

If `m` of `n` rows are potentially changed, then for fixed per-row/action utilities in `[0,1]`, the absolute change in their **unweighted mean** is at most `m/n`. This follows because certified rows contribute zero change and every other row contributes at most one in absolute value. It does not report observed utility change, say whether a change helps, or imply the base policy is good.

Real corrections may be coupled across actions or rows, or constrained by reachable probe histories. They may occupy only part of the declared box. Thus an uncertified row need not change in practice. The certificate remains valid for a smaller feasible set contained in the box; the set of potential winners can be conservative for that real router.

To justify bounds for a linear correction `w[j] · z`, a caller can use `sum(abs(w[j,k]) * c[k])` when every profile coordinate satisfies `abs(z[k]) <= c[k]`. Account separately for preprocessing, clipping, intercept changes, and numerical computation. An empirical maximum from sampled profiles alone is not a universal bound. This tool certifies the supplied rational model; it does not model floating-point errors in another implementation.

## Python and example

```python
from searchprobe.responsiveness import audit_response_json, audit_response_model

report = audit_response_json("model.json")
assert report["bound_validity_verified"] is False
# audit_response_model(model_dict) returns the same JSON-safe report.
```

From the repository root, without installation:

```sh
PYTHONPATH=packages/searchprobe/src python3 -m searchprobe response-audit \
  packages/searchprobe/examples/response_model.json --output /tmp/response-report.json
```

The bundled example is synthetic: one of three rows is certified unchanged, giving an absolute mean utility-change bound of `2/3`. A computed conditional certificate exits `0`, regardless of the unchanged fraction. Invalid input and file errors exit `2`. `--output` must differ from the input path. This command is separate from paired-log `audit` and labeled-source `decision-audit`.
