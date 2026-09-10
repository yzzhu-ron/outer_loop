"""Post-run replay-helper boundaries; the frozen research test suite is unchanged."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay_portable


class PortableHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.paper = self.base / "source"
        self.root = self.paper / "full_paper" / "intervention_transfer"
        self.root.mkdir(parents=True)
        code = self.root / "experiment.py"
        code.write_text("# Synthetic copied implementation\n")
        code_hash = hashlib.sha256(code.read_bytes()).hexdigest()
        self.write("freeze.v1.json", {"files": {"full_paper/intervention_transfer/experiment.py": code_hash}})
        self.write("inputs.v1.json", {"synthetic": True})
        self.write("decisions.v1.json", {replay_portable.TIMESTAMP: "original-time", "policy": 3})
        self.write("test_labels.v1.json", {"synthetic": True})
        self.write("results.v1.json", {"score": 0.5})
        self.calls = []
        self.change_policy = False

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, sort_keys=True) + "\n")

    def fake_run(self, command, cwd, **kwargs):
        # The subprocess changes cwd, so all command paths must remain absolute.
        self.assertTrue(Path(command[1]).is_absolute())
        self.assertTrue(Path(cwd).is_absolute())
        self.assertTrue(Path(command[-1]).is_absolute())
        self.calls.append(command[2])
        output = Path(command[-1])
        if command[2] == "decide":
            value = json.loads((Path(cwd) / "decisions.v1.json").read_text())
            value[replay_portable.TIMESTAMP] = "new-time"
            if self.change_policy:
                value["policy"] = 4
            output.write_text(json.dumps(value, sort_keys=True) + "\n")
        else:
            output.write_bytes((Path(cwd) / "results.v1.json").read_bytes())

    def run_copy(self, destination):
        with patch.object(replay_portable, "ROOT", self.root), patch.object(replay_portable, "PAPER", self.paper), \
                patch.object(replay_portable.subprocess, "run", side_effect=self.fake_run):
            return replay_portable.replay(destination)

    def test_relative_destination_resolves_before_subprocess_changes_directory(self):
        old_cwd = Path.cwd()
        try:
            os.chdir(self.base)
            report = self.run_copy(Path("relative-copy"))
        finally:
            os.chdir(old_cwd)
        self.assertEqual(Path(report["copy_directory"]), self.base / "relative-copy")
        self.assertEqual(self.calls, ["decide", "evaluate"])
        self.assertTrue(report["results_byte_identical"])
        self.assertFalse((self.base / "relative-copy" / "semantic_pilot" / "cache").exists())
        archived = json.loads((self.root / "decisions.v1.json").read_text())
        self.assertEqual(archived[replay_portable.TIMESTAMP], "original-time")
        self.assertNotEqual(report["archived_decision_sha256"], report["replayed_decision_sha256"])

    def test_existing_destination_is_rejected_without_overwrite(self):
        target = self.base / "existing"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_copy(target)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertEqual(self.calls, [])

    def test_non_timestamp_decision_changes_are_rejected(self):
        self.change_policy = True
        with self.assertRaisesRegex(AssertionError, "differ beyond"):
            self.run_copy(self.base / "different-policy")
        self.assertEqual(json.loads((self.root / "decisions.v1.json").read_text())["policy"], 3)


if __name__ == "__main__":
    unittest.main()
