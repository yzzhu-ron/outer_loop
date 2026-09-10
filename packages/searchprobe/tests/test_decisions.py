from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction as F
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from searchprobe.cli import main
from searchprobe.decisions import audit_decision_json, audit_decision_model


def source_model(utilities=None):
    utilities = utilities if utilities is not None else [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    return {
        "schema_version": 1, "kind": "source_utility_model", "model_id": "synthetic-test",
        "actions": [f"a{i}" for i in range(len(utilities[0]))],
        "world_ids": [f"w{i}" for i in range(len(utilities))], "utilities": utilities,
        "provenance": {"utility_source": "Synthetic test construction", "source_split": "No measured data",
                       "task_distribution": "One abstract decision per world", "synthetic": True},
        "assumptions": ["The supplied worlds define this synthetic model only."],
    }


class DecisionAuditTests(unittest.TestCase):
    def test_multiway_conflict_has_exact_primal_dual_certificate(self):
        report = audit_decision_model(source_model())
        self.assertEqual(report["decision_radius"]["exact"], "1/3")
        self.assertEqual(report["pairwise_sum_incompatibility"]["exact"], "0")
        self.assertEqual(report["certificate"]["duality_gap"]["exact"], "0")
        self.assertEqual(report["certificate"]["witness_support"], 3)
        self.assertEqual(report["certificate"]["primal_max_regret"], report["certificate"]["dual_min_expected_regret"])
        self.assertEqual(report["diagnostics"][0]["code"], "multiway_conflict")

    def test_unique_winner_conflict_retains_rational_precision(self):
        report = audit_decision_model(source_model([[0, 1, "99/100"], ["99/100", 0, 1], [1, "99/100", 0]]))
        self.assertEqual(report["decision_radius"]["exact"], "101/300")
        self.assertEqual(report["pairwise_sum_incompatibility"]["exact"], "1/100")
        self.assertEqual(report["model"]["utilities_exact"][0][2], "99/100")

    def test_two_actions_match_asymmetric_closed_form(self):
        report = audit_decision_model(source_model([[0, F(1, 5)], [F(7, 10), 0]]))
        self.assertEqual(report["decision_radius"]["exact"], "7/45")
        self.assertEqual(report["minimax_action_mixture"][1]["probability"]["exact"], "2/9")

    def test_zero_radius_does_not_imply_target_guarantee(self):
        report = audit_decision_model(source_model([[1, 0], [1, "1/2"]]))
        self.assertEqual(report["decision_radius"]["exact"], "0")
        self.assertEqual(report["common_optimal_actions"], ["a0"])
        self.assertIs(report["target_regret_guarantee"], False)
        self.assertIn("not a target-regret guarantee", report["scope"])
        self.assertTrue(any("zero does not certify" in item for item in report["required_interpretation"]))

    def test_single_action_and_tied_worlds_have_sparse_certificates(self):
        report = audit_decision_model(source_model([[0], [0.1], [0.8], [1]]))
        self.assertEqual(report["decision_radius"]["exact"], "0")
        self.assertEqual(report["certificate"]["witness_support"], 1)
        self.assertEqual(report["pairwise_sum_incompatibility"]["exact"], "0")

    def test_validation_rejects_empty_model_shape_errors_and_duplicate_names(self):
        for update in ({"world_ids": []}, {"actions": ["same", "same", "same"]},
                       {"world_ids": ["w", "w", "w"]}, {"utilities": [[0, 1]]},
                       {"utilities": [[1], [1], [1]]}, {"schema_version": True},
                       {"kind": "paired_probe_log"}, {"future_target_utilities": [1, 2]},
                       {"assumptions": []}):
            model = source_model()
            model.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                audit_decision_model(model)

    def test_provenance_is_required_even_for_numeric_matrix(self):
        for provenance in (None, {}, {"utility_source": "example"},
                           {"utility_source": "example", "source_split": "train", "task_distribution": "fixed", "synthetic": "yes"}):
            model = source_model()
            model["provenance"] = provenance
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                audit_decision_model(model)

    def test_nonfinite_bool_and_out_of_range_utilities_rejected(self):
        for value in (True, float("inf"), float("nan"), "1/0", "NaN", -1, "101/100", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit_decision_model(source_model([[value, 0], [0, 1]]))

    def test_exact_solver_work_limit_is_enforced_before_enumeration(self):
        with self.assertRaisesRegex(ValueError, "38 linear systems"):
            audit_decision_model(source_model(), max_vertex_systems=37)
        self.assertEqual(audit_decision_model(source_model(), max_vertex_systems=38)["method"]["vertex_systems_enumerated"], 38)
        for limit in (0, True, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                audit_decision_model(source_model(), max_vertex_systems=limit)

    def test_report_is_deterministic_json_safe_and_nonmutating(self):
        model = source_model()
        before = deepcopy(model)
        first = audit_decision_model(model)
        self.assertEqual(first, audit_decision_model(model))
        self.assertEqual(model, before)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)

    def test_json_loader_rejects_duplicate_keys_and_nonjson_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for text in ('{"kind":1,"kind":2}', '{"utilities":NaN}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    audit_decision_json(path)

    def test_decision_cli_is_separate_and_reports_its_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "model.json", Path(directory) / "report.json"
            path.write_text(json.dumps(source_model()), encoding="utf-8")
            with redirect_stderr(StringIO()) as stderr, redirect_stdout(StringIO()) as stdout:
                result = main(["decision-audit", str(path), "--output", str(output)])
            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("not a target-regret guarantee", stderr.getvalue())
            report = json.loads(output.read_text())
            self.assertEqual(report["audit_kind"], "source_model_decision_radius")
            self.assertNotIn("counts", report)

    def test_cli_refuses_paired_probe_log_as_a_source_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text('{"probe_id":"p","family":"f"}', encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                main(["decision-audit", str(path)])
            self.assertEqual(caught.exception.code, 2)

    def test_cli_refuses_overwriting_source_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            original = json.dumps(source_model())
            path.write_text(original, encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(["decision-audit", str(path), "--output", str(path)])
            self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
