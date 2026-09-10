from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction as F
from io import StringIO
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from searchprobe.cli import main
from searchprobe.responsiveness import audit_response_json, audit_response_model


def score_model(scores=None, bounds=None):
    scores = scores if scores is not None else [["0.8", "0.3"], ["0.4", "0.5"], ["0.49", "0.5"]]
    return {
        "schema_version": 1, "kind": "score_box_model", "model_id": "synthetic-test",
        "actions": [f"a{i}" for i in range(len(scores[0]))],
        "row_ids": [f"r{i}" for i in range(len(scores))], "base_scores": scores,
        "correction_bounds": bounds if bounds is not None else ["0.05", "0.05"],
        "provenance": {"score_source": "Synthetic test scores", "bound_source": "Assumed test bounds",
                       "row_population": "Synthetic rows", "synthetic": True},
        "assumptions": ["All additive corrections lie in the declared independent score box."],
    }


class ResponsivenessAuditTests(unittest.TestCase):
    def test_two_action_stability_and_utility_bound(self):
        report = audit_response_model(score_model())
        self.assertEqual(report["counts"], {"rows": 3, "actions": 2, "certified_unchanged": 1, "potentially_changed": 2})
        self.assertEqual([row["certified_unchanged"] for row in report["rows"]], [True, False, False])
        self.assertEqual(report["unchanged_fraction"]["exact"], "1/3")
        self.assertEqual(report["mean_utility_change_bound"]["signed_interval_exact"], ["-2/3", "2/3"])
        self.assertEqual(report["mean_utility_change_bound"]["absolute_upper_bound"]["exact"], "2/3")

    def test_ties_respect_action_order_in_both_directions(self):
        report = audit_response_model(score_model([["0.5", "0.4"], ["0.4", "0.5"]]))
        first, second = report["rows"]
        self.assertEqual(first["original_action_index"], 0)
        self.assertTrue(first["certified_unchanged"])
        self.assertEqual(first["potential_challenger_indices"], [])
        self.assertEqual(second["original_action_index"], 1)
        self.assertFalse(second["certified_unchanged"])
        self.assertEqual(second["potential_challenger_indices"], [0])

    def test_zero_bound_base_tie_is_fixed(self):
        row = audit_response_model(score_model([[1, 1, 1]], [0, 0, 0]))["rows"][0]
        self.assertEqual(row["original_action_index"], 0)
        self.assertTrue(row["certified_unchanged"])
        self.assertEqual(row["potential_challengers"], [])

    def test_original_winners_own_correction_can_change_decision(self):
        row = audit_response_model(score_model([[1, "0.9"]], ["0.2", 0]))["rows"][0]
        self.assertFalse(row["certified_unchanged"])
        self.assertEqual(row["winner_lower_score"]["exact"], "4/5")
        self.assertEqual(row["potential_challengers"], ["a1"])

    def test_third_action_can_block_a_pairwise_threat(self):
        # a2 can beat the original winner's low score, but never a1's low score.
        row = audit_response_model(score_model([[1, "0.9", "0.8"]], [1, 0, "0.05"]))["rows"][0]
        self.assertFalse(row["certified_unchanged"])
        self.assertEqual(row["potential_challenger_indices"], [1])

    def test_single_action_is_unchanged_for_any_finite_bound(self):
        report = audit_response_model(score_model([[-10], ["1/3"]], ["1e400"]))
        self.assertEqual(report["counts"]["certified_unchanged"], 2)
        self.assertEqual(report["mean_utility_change_bound"]["absolute_upper_bound"]["exact"], "0")

    def test_claims_match_independent_corner_enumeration(self):
        # Every feasible winner has a corner witness: its own upper score and
        # every competitor's lower score. Enumerate all corners independently.
        for scores in product((-1, 0, 1), repeat=3):
            for bounds in product((0, 1), repeat=3):
                original = max(range(3), key=lambda j: scores[j])
                feasible = set()
                for signs in product((-1, 1), repeat=3):
                    corner = [scores[j] + signs[j] * bounds[j] for j in range(3)]
                    feasible.add(max(range(3), key=lambda j: corner[j]))
                row = audit_response_model(score_model([scores], bounds))["rows"][0]
                with self.subTest(scores=scores, bounds=bounds):
                    self.assertEqual(row["certified_unchanged"], feasible == {original})
                    self.assertEqual(row["potential_challenger_indices"], sorted(feasible - {original}))

    def test_translation_and_positive_scaling_preserve_decisions(self):
        model = score_model([[F(1, 3), F(1, 4), -1], [0, 0, 0]], [F(1, 10), 0, F(1, 2)])
        transformed = deepcopy(model)
        transformed["base_scores"] = [[17 * score + 1000 for score in row] for row in model["base_scores"]]
        transformed["correction_bounds"] = [17 * bound for bound in model["correction_bounds"]]
        before, after = audit_response_model(model), audit_response_model(transformed)
        self.assertEqual(before["counts"], after["counts"])
        for left, right in zip(before["rows"], after["rows"]):
            for key in ("original_action_index", "certified_unchanged", "potential_challenger_indices"):
                self.assertEqual(left[key], right[key])

    def test_decimal_fraction_and_float_inputs_have_declared_precision(self):
        report = audit_response_model(score_model([[Decimal("0.1"), F(1, 3), 0.2]], [0, 0, 0]))
        self.assertEqual(report["model"]["base_scores_exact"], [["1/10", "1/3", "1/5"]])
        self.assertEqual(report["rows"][0]["original_action_index"], 1)

    def test_nonfinite_bool_and_malformed_numbers_rejected(self):
        for value in (True, False, float("inf"), float("nan"), Decimal("NaN"), "Infinity", "1/0", "bad", None):
            for field in ("base_scores", "correction_bounds"):
                model = score_model()
                if field == "base_scores":
                    model[field][0][0] = value
                else:
                    model[field][0] = value
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    audit_response_model(model)

    def test_negative_or_missing_correction_bounds_rejected(self):
        for bounds in (None, [-1, 0], [0], [[0, 0]], "0.1"):
            model = score_model()
            model["correction_bounds"] = bounds
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                audit_response_model(model)
        model = score_model()
        del model["correction_bounds"]
        with self.assertRaises(ValueError):
            audit_response_model(model)

    def test_empty_shapes_duplicate_ids_and_unknown_fields_rejected(self):
        for update in ({"actions": []}, {"row_ids": []}, {"base_scores": []},
                       {"actions": ["same", "same"]}, {"row_ids": ["r", "r", "r"]},
                       {"base_scores": [[0, 1]]}, {"base_scores": [[1], [1], [1]]},
                       {"schema_version": True}, {"kind": "paired_probe_log"},
                       {"model_id": " "}, {"target_labels": [0, 1]}, {"assumptions": []}):
            model = score_model()
            model.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                audit_response_model(model)
        for model in (None, [], {}):
            with self.subTest(model=model), self.assertRaises(ValueError):
                audit_response_model(model)

    def test_provenance_is_required_and_explicit(self):
        for provenance in (None, {}, {"score_source": "test"},
                           dict(score_model()["provenance"], synthetic="yes"),
                           dict(score_model()["provenance"], bound_source=" "),
                           dict(score_model()["provenance"], extra="unrecognized")):
            model = score_model()
            model["provenance"] = provenance
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                audit_response_model(model)

    def test_report_is_deterministic_json_safe_nonmutating_and_conditional(self):
        model = score_model()
        before = deepcopy(model)
        report = audit_response_model(model)
        self.assertEqual(report, audit_response_model(model))
        self.assertEqual(model, before)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        for key in ("bound_validity_verified", "target_regret_guarantee", "generalization_guarantee", "labels_or_outcomes_used"):
            self.assertIs(report[key], False)

    def test_finite_scores_beyond_float_range_retain_exact_value(self):
        report = audit_response_model(score_model([["1e400", 0]], [0, 0]))
        self.assertEqual(report["rows"][0]["winner_lower_score"], {"exact": str(10 ** 400), "value": None})
        json.dumps(report, allow_nan=False)

    def test_json_loader_preserves_decimal_precision_at_tie_boundary(self):
        model = score_model([["DECIMAL_TOKEN", "0.1"]], [0, 0])
        literal = "0.1000000000000000000000000000000000001"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            # Put the more precise score at index 1 so float rounding would
            # incorrectly route to index 0 under the tie rule.
            model["base_scores"][0].reverse()
            path.write_text(json.dumps(model).replace('"DECIMAL_TOKEN"', literal), encoding="utf-8")
            report = audit_response_json(path)
        self.assertEqual(report["rows"][0]["original_action_index"], 1)
        self.assertEqual(report["model"]["base_scores_exact"][0][1], str(F(literal)))

    def test_json_loader_rejects_duplicate_keys_nonjson_numbers_and_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for body in ('{"kind":1,"kind":2}', '{"base_scores":NaN}', '{"base_scores":Infinity}', ""):
                path.write_text(body, encoding="utf-8")
                with self.subTest(body=body), self.assertRaises(ValueError):
                    audit_response_json(path)

    def test_cli_example_writes_report_and_conditional_scope(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "response_model.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with redirect_stderr(StringIO()) as stderr, redirect_stdout(StringIO()) as stdout:
                status = main(["response-audit", str(example), "--output", str(output)])
            report = json.loads(output.read_text())
        self.assertEqual(status, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("1/3 rows certified unchanged", stderr.getvalue())
        self.assertIn("supplied bounds are not verified", stderr.getvalue())
        self.assertEqual(report["audit_kind"], "conditional_score_box_responsiveness")
        self.assertEqual(report["potentially_changed_fraction"]["exact"], "2/3")

    def test_cli_stdout_and_invalid_model_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(score_model()), encoding="utf-8")
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()) as stdout:
                self.assertEqual(main(["response-audit", str(path)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["counts"]["rows"], 3)
            path.write_text('{"probe_id":"p"}', encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                main(["response-audit", str(path)])
            self.assertEqual(caught.exception.code, 2)

    def test_cli_refuses_overwriting_input_or_its_link(self):
        with tempfile.TemporaryDirectory() as directory:
            path, link = Path(directory) / "model.json", Path(directory) / "alias.json"
            original = json.dumps(score_model())
            path.write_text(original, encoding="utf-8")
            link.symlink_to(path)
            for output in (path, link):
                with self.subTest(output=output), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                    main(["response-audit", str(path), "--output", str(output)])
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
