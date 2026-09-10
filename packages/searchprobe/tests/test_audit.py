from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from searchprobe import audit_jsonl, audit_records, canonical_query, validate_record


def probe(probe_id="p1", **changes):
    record = {"probe_id": probe_id, "family": "indirect", "actions": ["original", "rewrite"],
              "queries": ["how do I file income taxes", "income tax filing instructions"],
              "cost": {"values": [1, 1], "total": 2, "unit": "search_calls"}}
    record.update(changes)
    return record


def codes(report):
    return {diagnostic["code"] for diagnostic in report["diagnostics"]}


class ValidationTests(unittest.TestCase):
    def test_valid_preflight_has_no_observed_endpoint(self):
        record = probe(cost={"values": [0, 0], "total": 0, "unit": "search_calls"})
        self.assertEqual(validate_record(record), [])
        report = audit_records([record], min_per_cell=1)
        self.assertEqual(report["counts"]["observed_records"], 0)
        self.assertIsNone(report["rates"]["zero_contrast_among_observed"])
        self.assertEqual(report["status"], "ok")

    def test_bad_shape_and_unknown_fields(self):
        for record in (None, [], {}, probe(queries=["one"]), probe(queries=["  ", "two"]),
                       probe(actions=["same", "same"]), probe(bucket=4), probe(future_task_label=1)):
            with self.subTest(record=record):
                self.assertTrue(validate_record(record))

    def test_invalid_ranks_and_cutoffs(self):
        for ranks, cutoff in (([0, 1], 10), ([True, 1], 10), ([11, 1], 10), ([1.5, None], 10),
                              ([1, 1], True), ([1, 1], 0), ([1], 10)):
            with self.subTest(ranks=ranks, cutoff=cutoff):
                self.assertTrue(validate_record(probe(known_target={"ranks": ranks, "cutoff": cutoff})))

    def test_null_is_valid_censored_absence(self):
        self.assertEqual(validate_record(probe(known_target={"id": "doc", "ranks": [None, 10], "cutoff": 10})), [])

    def test_cost_errors_are_accounting_violations(self):
        for cost in (None, {"values": [1, 1], "total": 1, "unit": "calls"},
                     {"values": [-1, 1], "total": 0, "unit": "calls"},
                     {"values": [True, 1], "total": 2, "unit": "calls"},
                     {"values": [float("nan"), 1], "total": 2, "unit": "calls"},
                     {"values": [1, 1], "total": float("inf"), "unit": "calls"},
                     {"values": [1, 1], "total": 2, "unit": ""}):
            with self.subTest(cost=cost):
                self.assertIn("accounting_violation", {item["code"] for item in validate_record(probe(cost=cost))})

    def test_normal_float_rounding_is_not_accounting_error(self):
        self.assertEqual(validate_record(probe(cost={"values": [0.1, 0.2], "total": 0.3, "unit": "USD"})), [])

    def test_outcome_bounds_and_nonfinite_values_are_rejected(self):
        for outcomes in ({"values": [0, 2], "bounds": [0, 1]}, {"values": [1, 1], "bounds": [1, 1]},
                         {"values": [1, float("inf")]}, {"values": [0, 1], "higher_is_better": "yes"}):
            with self.subTest(outcomes=outcomes):
                self.assertTrue(validate_record(probe(outcomes=outcomes)))

    def test_endpoint_must_be_unambiguous(self):
        self.assertTrue(validate_record(probe(known_target={"ranks": [1, 1], "cutoff": 10}, outcomes={"values": [0, 1]})))

    def test_huge_integer_is_rejected_without_crash(self):
        self.assertTrue(validate_record(probe(outcomes={"values": [10**1000, 1]})))


class AuditTests(unittest.TestCase):
    def test_canonicalization_preserves_multiplicity_and_unicode(self):
        self.assertEqual(canonical_query("Ｃafé TAX tax!"), canonical_query("tax café tax"))
        self.assertNotEqual(canonical_query("tax tax"), canonical_query("tax"))

    def test_exact_and_bag_collisions_are_distinguished(self):
        report = audit_records([
            probe("exact", queries=["tax help", "tax help"]),
            probe("bag", queries=["Tax help!", "help tax"]),
            probe("different", queries=["tax help", "income filing"]),
        ], min_per_cell=1)
        self.assertEqual(report["counts"]["exact_identical_query_pairs"], 1)
        self.assertEqual(report["counts"]["token_bag_identical_query_pairs"], 2)
        self.assertEqual(report["counts"]["token_bag_distinct_query_pairs"], 1)
        self.assertTrue({"exact_query_collision", "token_bag_collision"} <= codes(report))

    def test_saturation_floor_and_contrast_at_declared_cutoff(self):
        report = audit_records([
            probe("ceiling", known_target={"ranks": [1, 1], "cutoff": 10}),
            probe("floor", known_target={"ranks": [None, None], "cutoff": 10}),
            probe("tie", known_target={"ranks": [3, 3], "cutoff": 10}),
            probe("contrast", known_target={"ranks": [None, 10], "cutoff": 10}),
        ], min_per_cell=1)
        self.assertEqual(report["counts"]["saturated_success_pairs"], 1)
        self.assertEqual(report["counts"]["floor_pairs"], 1)
        self.assertEqual(report["counts"]["zero_contrast_pairs"], 3)
        self.assertEqual(report["counts"]["nonzero_contrast_pairs"], 1)
        self.assertEqual(report["rates"]["zero_contrast_among_observed"], 0.75)

    def test_lower_is_better_respects_bounds(self):
        report = audit_records([
            probe("good", outcomes={"values": [0, 0], "bounds": [0, 5], "higher_is_better": False}),
            probe("bad", outcomes={"values": [5, 5], "bounds": [0, 5], "higher_is_better": False}),
        ], min_per_cell=1)
        examples = {d["code"]: d.get("example_probe_ids") for d in report["diagnostics"]}
        self.assertEqual(examples["saturated_success"], ["good"])
        self.assertEqual(examples["observed_floor"], ["bad"])

    def test_unbounded_endpoints_do_not_infer_saturation(self):
        report = audit_records([probe(outcomes={"values": [1, 1]})], min_per_cell=1)
        self.assertEqual(report["counts"]["zero_contrast_pairs"], 1)
        self.assertEqual(report["counts"]["saturated_success_pairs"], 0)
        self.assertEqual(report["counts"]["floor_pairs"], 0)

    def test_narrow_bounds_cannot_be_both_best_and_worst(self):
        report = audit_records([probe(outcomes={"values": [1e-15, 1e-15], "bounds": [0, 1e-15]})], min_per_cell=1)
        self.assertEqual(report["counts"]["saturated_success_pairs"], 1)
        self.assertEqual(report["counts"]["floor_pairs"], 0)

    def test_changed_outcome_for_same_query_checks_determinism(self):
        report = audit_records([probe(queries=["same", "same"], outcomes={"values": [0, 1]})], min_per_cell=1)
        self.assertIn("same_query_different_outcomes", codes(report))
        self.assertEqual(report["counts"]["invalid_records"], 0)

    def test_missing_expected_cells_and_unordered_action_pairs(self):
        report = audit_records([
            probe("one", actions=["a", "b"], bucket="easy"),
            probe("two", actions=["b", "a"], bucket="easy"),
        ], expected_action_pairs=[["a", "b"], ["a", "c"]], expected_buckets=["easy", "hard"], min_per_cell=2)
        cells = report["coverage"]["cells"]
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0]["records"], 2)
        self.assertEqual(report["coverage"]["sparse_cells"], 3)

    def test_unbucketed_records_remain_separate_from_named_bucket(self):
        report = audit_records([probe("one"), probe("two", bucket="unbucketed")], min_per_cell=1)
        self.assertEqual({cell["bucket"] for cell in report["coverage"]["cells"]}, {None, "unbucketed"})

    def test_collisions_reduce_distinct_coverage_without_changing_raw_coverage(self):
        report = audit_records([probe(str(i), queries=["same", "same"]) for i in range(3)])
        self.assertEqual(report["coverage"]["sparse_cells"], 0)
        self.assertEqual(report["coverage"]["sparse_distinct_query_cells"], 1)
        self.assertIn("sparse_distinct_query_coverage", codes(report))

    def test_invalid_and_duplicate_records_do_not_inflate_costs(self):
        report = audit_records([probe("one"), probe("one"), probe("bad", cost={"values": [1, 1], "total": 5, "unit": "search_calls"})])
        self.assertEqual(report["counts"]["valid_records"], 1)
        self.assertEqual(report["counts"]["invalid_records"], 2)
        self.assertEqual(report["counts"]["accounting_violation_records"], 1)
        self.assertEqual(report["costs_by_unit"]["search_calls"]["total"], 2)
        self.assertEqual(report["status"], "error")

    def test_cost_units_stay_separate_and_categories_overlap(self):
        report = audit_records([
            probe("one", queries=["same", "same"], outcomes={"values": [1, 1]}),
            probe("two", cost={"values": [2, 3], "total": 5, "unit": "milliseconds"}),
        ], min_per_cell=1)
        self.assertEqual(report["costs_by_unit"]["search_calls"], {"total": 2, "on_exact_identical_queries": 2, "on_zero_contrast_outcomes": 2})
        self.assertEqual(report["costs_by_unit"]["milliseconds"]["total"], 5)

    def test_cost_aggregate_overflow_is_a_serializable_error(self):
        records = [probe(str(i), cost={"values": [1e308, 0], "total": 1e308, "unit": "tiny_units"}) for i in range(2)]
        report = audit_records(records, min_per_cell=1)
        self.assertIn("cost_aggregation_overflow", codes(report))
        json.dumps(report, allow_nan=False)

    def test_empty_log_is_not_a_pass(self):
        report = audit_records([], expected_action_pairs=[["a", "b"]], expected_buckets=["hard"])
        self.assertEqual(report["status"], "error")
        self.assertIn("no_valid_probes", codes(report))

    def test_invalid_config_does_not_silently_weaken_checks(self):
        for kwargs in ({"min_per_cell": 0}, {"min_per_cell": True}, {"expected_action_pairs": [["a", "a"]]},
                       {"expected_buckets": "easy"}, {"expected_buckets": [None]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit_records([], **kwargs)

    def test_deterministic_and_nonmutating(self):
        records = [probe(str(i), bucket="easy") for i in range(4)]
        before = deepcopy(records)
        first = audit_records(records)
        self.assertEqual(first, audit_records(iter(records)))
        self.assertEqual(records, before)

    def test_contrast_does_not_claim_alignment(self):
        report = audit_records([probe(outcomes={"values": [0.1, 0.9], "bounds": [0, 1]})], min_per_cell=1)
        self.assertEqual(report["status"], "ok")
        self.assertIn("future-task", " ".join(report["what_this_log_cannot_establish"]))


class JsonlTests(unittest.TestCase):
    def test_bad_json_duplicate_keys_and_nonjson_numbers_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("\n" + json.dumps(probe()) + '\n{bad json\n{"x":1,"x":2}\n{"x":NaN}\n', encoding="utf-8")
            report = audit_jsonl(path, min_per_cell=1)
        self.assertEqual(report["counts"]["records"], 4)
        self.assertEqual(report["counts"]["valid_records"], 1)
        self.assertEqual([item["line"] for item in report["invalid_records"]], [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
