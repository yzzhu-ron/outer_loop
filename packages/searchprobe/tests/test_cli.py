from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from searchprobe.cli import main


RECORD = {"probe_id": "p", "family": "synthetic", "actions": ["a", "b"], "queries": ["same", "same"],
          "cost": {"values": [0, 0], "total": 0, "unit": "search_calls"}}


class CliTests(unittest.TestCase):
    def test_cli_writes_report_and_can_fail_on_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "log.jsonl", Path(directory) / "report.json"
            source.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()) as stdout:
                result = main(["audit", str(source), "--output", str(output), "--fail-on-warnings"])
            self.assertEqual(result, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(json.loads(output.read_text())["counts"]["exact_identical_query_pairs"], 1)

    def test_stdout_contains_only_json_and_default_warnings_are_not_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "log.jsonl"
            source.write_text(json.dumps(RECORD), encoding="utf-8")
            with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
                result = main(["audit", str(source), "--expect-pair", "a", "c", "--expect-bucket", "hard"])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "warning")
            self.assertIn("searchprobe: warning", stderr.getvalue())

    def test_invalid_records_still_produce_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "log.jsonl"
            source.write_text("not json\n", encoding="utf-8")
            with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()):
                result = main(["audit", str(source)])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(stdout.getvalue())["counts"]["invalid_records"], 1)

    def test_refuses_overwriting_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "log.jsonl"
            original = json.dumps(RECORD)
            source.write_text(original, encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                main(["audit", str(source), "--output", str(source)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(source.read_text(), original)


if __name__ == "__main__":
    unittest.main()
