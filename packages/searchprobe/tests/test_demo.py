from contextlib import redirect_stdout, redirect_stderr
from importlib.resources import as_file, files
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from searchprobe.cli import main as audit_main
from searchprobe.demo import EXAMPLES, main, run_demo


class DemoTests(unittest.TestCase):
    def test_bundled_resources_match_the_reviewed_repository_examples(self):
        original = Path(__file__).resolve().parents[1] / "examples"
        for filename, _, _ in EXAMPLES.values():
            with self.subTest(filename=filename), as_file(files("searchprobe").joinpath("example_data", filename)) as packaged:
                self.assertEqual(packaged.read_bytes(), (original / filename).read_bytes())

    def test_all_demos_export_inputs_and_compute_expected_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "examples"
            summary = run_demo(output)
            self.assertTrue(summary["synthetic"])
            self.assertEqual(len(summary["examples"]), 3)
            probes = json.loads((output / "probes.report.json").read_text())
            self.assertEqual(probes["counts"]["valid_records"], 5)
            self.assertEqual(probes["counts"]["exact_identical_query_pairs"], 1)
            decision = json.loads((output / "decision.report.json").read_text())
            self.assertEqual(decision["decision_radius"]["exact"], "1/3")
            response = json.loads((output / "response.report.json").read_text())
            self.assertEqual(response["unchanged_fraction"]["exact"], "1/3")
            self.assertEqual(response["mean_utility_change_bound"]["absolute_upper_bound"]["exact"], "2/3")
            self.assertFalse(response["bound_validity_verified"])
            self.assertIn("No retrieval", (output / "README.md").read_text())

    def test_exported_inputs_rerun_through_all_original_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = run_demo(output)
            for item in summary["examples"]:
                with self.subTest(name=item["name"]), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    target = output / (item["name"] + ".rerun.json")
                    self.assertEqual(audit_main([item["command"][1], str(output / item["input"]), "--output", str(target)]), 0)
                    self.assertEqual(json.loads(target.read_text()), json.loads((output / item["report"]).read_text()))

    def test_single_example_and_module_cli(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as stdout:
            output = Path(directory) / "only-response"
            self.assertEqual(main(["response", "--output-dir", str(output)]), 0)
            self.assertIn("Created 1 synthetic", stdout.getvalue())
            self.assertTrue((output / "response.report.json").exists())
            self.assertFalse((output / "probes.report.json").exists())

    def test_existing_or_symlinked_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            target = output / "kept.json"
            target.write_text("keep me")
            (output / "response_model.json").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                run_demo(output)
            self.assertEqual(target.read_text(), "keep me")
            self.assertFalse((output / "demo.json").exists())

    def test_invalid_example_has_no_file_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with self.assertRaises(ValueError):
                run_demo(output, "unknown")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
