from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_trace_exports import convert_trace, export_traces


def record(probe_id="one", action="semantic"):
    requests = 2 if action == "decomposed" else 1
    return {"probe_id": probe_id, "family": "indirect_question", "actions": ["original", action],
            "queries": ["tax obligations", "income tax\nfiling guidance"],
            "known_target": {"ranks": [None, 4], "cutoff": 10}, "bucket": "2",
            "cost": {"values": [1, requests], "total": 1 + requests, "unit": "search_calls"}}


class TraceExportTests(unittest.TestCase):
    def test_cli_export_preserves_raw_data_and_records_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            raw_dir, raw_report_dir = results / "probe_traces", results / "probe_diagnostics"
            raw_dir.mkdir()
            raw_report_dir.mkdir()
            raw = raw_dir / "example_11.jsonl"
            original = "\n".join(json.dumps(x) for x in [record(), record("two", "decomposed")]) + "\n"
            raw.write_text(original)
            raw_report = raw_report_dir / "example_11.json"
            raw_report.write_text('{"original":true}')
            with redirect_stdout(StringIO()):
                provenance = export_traces(results)
            self.assertEqual(raw.read_text(), original)
            self.assertEqual(raw_report.read_text(), '{"original":true}')
            self.assertEqual(provenance["counts"]["raw_records"], 2)
            self.assertEqual(provenance["counts"]["retained_records"], 1)
            self.assertEqual(provenance["counts"]["excluded_records"], 1)
            strict = json.loads((results / "strict_probe_traces" / raw.name).read_text())
            self.assertEqual(strict["queries"], record()["queries"])
            self.assertEqual(strict["known_target"], record()["known_target"])
            self.assertEqual(strict["cost"]["unit"], "nominal_selected_probe_search_calls")
            report = json.loads((results / "strict_probe_diagnostics" / "example_11.json").read_text())
            self.assertEqual(report["counts"]["valid_records"], 1)
            self.assertIn("nominal_selected_probe_search_calls", report["costs_by_unit"])
            self.assertTrue((results / "trace_provenance.json").exists())

    def test_missing_traces_do_not_create_a_fake_success(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "run retrieval first"):
            export_traces(directory)

    def test_unexpected_action_cost_or_unit_fails_closed(self):
        examples = [record(action="unknown"), record(), record()]
        examples[1]["cost"]["unit"] = "USD"
        examples[2]["cost"] = {"values": [1, 2], "total": 3, "unit": "search_calls"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            for example in examples:
                path.write_text(json.dumps(example))
                with self.subTest(example=example), self.assertRaises(ValueError):
                    convert_trace(path)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text('{"probe_id":"one","probe_id":"two"}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                convert_trace(path)

    def test_duplicate_probe_ids_are_caught_by_the_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "probe_traces").mkdir()
            (results / "probe_traces" / "example.jsonl").write_text((json.dumps(record()) + "\n") * 2)
            with self.assertRaisesRegex(RuntimeError, "CLI audit failed"):
                export_traces(results)

    def test_only_decomposed_pairs_produce_explicit_no_evidence_report(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "probe_traces").mkdir()
            (results / "probe_traces" / "example.jsonl").write_text(json.dumps(record(action="decomposed")))
            with redirect_stdout(StringIO()):
                provenance = export_traces(results)
            self.assertEqual(provenance["counts"]["files_without_retained_pairs"], 1)
            self.assertEqual(provenance["files"][0]["strict_audit_exit_code"], 2)
            self.assertEqual(provenance["files"][0]["strict_audit_status"], "error")

    def test_derived_symlink_cannot_overwrite_raw_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            raw_dir = results / "probe_traces"
            raw_dir.mkdir()
            raw = raw_dir / "example.jsonl"
            original = json.dumps(record())
            raw.write_text(original)
            (results / "strict_probe_traces").symlink_to(raw_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "protected raw directory"):
                export_traces(results)
            self.assertEqual(raw.read_text(), original)


if __name__ == "__main__":
    unittest.main()
