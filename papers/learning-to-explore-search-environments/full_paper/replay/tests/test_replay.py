"""Replay the committed fixture and reject coherent and incoherent corruption."""

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1]
EVIDENCE = HERE.parents[1] / "semantic_pilot" / "results"
spec = importlib.util.spec_from_file_location("portable_replay", HERE / "replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class PortableReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "evidence"
        self.root.mkdir()
        self.manifest_path = Path(self.temporary.name) / "manifest.json"
        self.manifest = replay.load_manifest(HERE / "evidence_manifest.json")
        for name in self.manifest["files"]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EVIDENCE / name, target)
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")

    def refresh(self, name):
        self.manifest["files"][name] = replay.sha256(self.root / name)
        self.write_manifest()

    def change_json(self, name, mutate):
        path = self.root / name
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value) + "\n")
        self.refresh(name)

    def run_replay(self):
        return replay.replay(self.root, self.manifest_path)

    def test_committed_evidence_replays_in_a_cache_free_directory_without_writes(self):
        before = {name: replay.sha256(self.root / name) for name in self.manifest["files"]}
        with patch.object(replay.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network")):
            report = self.run_replay()
        self.assertEqual(report["counts"], {"environments": 6, "query_backend_rows": 730,
            "main_run_rows_recomputed": 450, "followup_run_rows_recomputed": 360,
            "followup_baseline_rows_recomputed": 18, "paired_contrast_means_recomputed": 600,
            "response_certificates_recomputed": 6, "source_decision_certificates_recomputed": 6})
        self.assertEqual(report["input_files_verified"], 34)
        self.assertEqual(sum(row["certified_unchanged"] for row in report["certificates"]), 730)
        scifact = next(row for row in report["environments"] if row["environment"] == "scifact_dense")
        self.assertAlmostEqual(scifact["followup_prior_ndcg"], 0.6376, places=4)
        self.assertAlmostEqual(scifact["followup_decision_B32_ndcg"], 0.6795, places=4)
        self.assertTrue(report["uses_recorded_target_outcomes"])
        self.assertFalse(report["uses_target_outcomes_for_fitting_or_selection"])
        self.assertTrue(any("Bootstrap sampling" in item for item in report["not_replayed"]))
        self.assertEqual({name: replay.sha256(self.root / name) for name in self.manifest["files"]}, before)
        self.assertFalse((self.root / "cache").exists())

    def test_byte_drift_is_rejected_before_statistics_are_calculated(self):
        path = self.root / "headroom.csv"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.run_replay()

    def test_summary_error_is_rejected_even_after_refreshing_its_checksum(self):
        path = self.root / "headroom.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["source_router"] = "0.99"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.refresh("headroom.csv")
        with self.assertRaisesRegex(ValueError, "Recomputed value differs"):
            self.run_replay()

    def test_missing_run_and_invalid_action_are_rejected(self):
        self.change_json("paired_outcomes.json", lambda data: data["fiqa_bm25"]["runs"].pop("11/random/4"))
        with self.assertRaisesRegex(ValueError, "run grid mismatch"):
            self.run_replay()
        shutil.copyfile(EVIDENCE / "paired_outcomes.json", self.root / "paired_outcomes.json")
        self.change_json("paired_outcomes.json", lambda data: data["fiqa_bm25"]["runs"]["11/random/4"]["actions"].__setitem__(0, True))
        with self.assertRaisesRegex(ValueError, "Invalid actions"):
            self.run_replay()

    def test_followup_query_order_and_selected_utilities_are_checked(self):
        self.change_json("world_model_followup/paired_outcomes.json", lambda data: data["fiqa_bm25"]["query_ids"].reverse())
        with self.assertRaisesRegex(ValueError, "cohort/order"):
            self.run_replay()
        shutil.copyfile(EVIDENCE / "world_model_followup/paired_outcomes.json", self.root / "world_model_followup/paired_outcomes.json")
        self.change_json("world_model_followup/paired_outcomes.json", lambda data: data["fiqa_bm25"]["runs"]["11/random/4"]["ndcg"].__setitem__(0, 0.123456))
        with self.assertRaisesRegex(ValueError, "Recomputed value differs"):
            self.run_replay()

    def test_exact_certificate_is_recomputed_after_report_hash_update(self):
        self.change_json("response_models/fiqa_bm25.audit.json", lambda data: data["counts"].__setitem__("certified_unchanged", 0))
        with self.assertRaisesRegex(ValueError, "Response API/report mismatch"):
            self.run_replay()

    def test_incomplete_manifest_and_duplicate_csv_grid_are_rejected(self):
        self.manifest["files"].pop("headroom.csv")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "complete six-environment"):
            self.run_replay()
        self.manifest = replay.load_manifest(HERE / "evidence_manifest.json")
        path = self.root / "adaptation.csv"
        lines = path.read_text().splitlines()
        lines[-1] = lines[1]
        path.write_text("\n".join(lines) + "\n")
        self.refresh("adaptation.csv")
        with self.assertRaisesRegex(ValueError, "main adaptation grid"):
            self.run_replay()

    def test_nonfinite_utilities_and_duplicate_json_keys_are_rejected(self):
        self.change_json("paired_outcomes.json", lambda data: data["fiqa_bm25"]["action_scores"][0].__setitem__(0, float("nan")))
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
            self.run_replay()
        path = self.root / "headroom-extra.json"
        path.write_text('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            replay.read_json(path)

    def test_download_skips_valid_files_and_checks_new_bytes_before_keeping_them(self):
        with patch.object(replay.urllib.request, "urlopen", side_effect=AssertionError("Unexpected download")):
            replay.download_evidence(self.root, self.manifest)
        path = self.root / "headroom.csv"
        body = path.read_bytes()
        path.unlink()
        with patch.object(replay.urllib.request, "urlopen", return_value=BytesIO(b"corrupt")):
            with self.assertRaisesRegex(ValueError, "Downloaded checksum differs"):
                replay.download_evidence(self.root, self.manifest)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob("*.download")), [])
        with patch.object(replay.urllib.request, "urlopen", return_value=BytesIO(body)) as download:
            replay.download_evidence(self.root, self.manifest)
        self.assertIn(self.manifest["commit"], download.call_args.args[0])
        self.assertEqual(path.read_bytes(), body)

    def test_cli_refuses_to_overwrite_an_evidence_file(self):
        path = self.root / "headroom.csv"
        before = path.read_bytes()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            replay.main(["--evidence-dir", str(self.root), "--manifest", str(self.manifest_path), "--output", str(path)])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_output_is_a_complete_json_report(self):
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            self.assertEqual(replay.main(["--evidence-dir", str(self.root), "--manifest", str(self.manifest_path)]), 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "replayed")
        self.assertIn("No retrieval, training, or bootstrap rerun", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
