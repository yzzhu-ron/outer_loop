"""Offline runner isolation, failure-cost, integrity, and input preparation tests."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("semantic_runner_offline_test", ROOT / "run_semantic.py")
downloader = load_module("semantic_downloader_offline_test", ROOT / "prepare_data.py")


def generated_row(*, valid=True):
    return {
        "valid": valid,
        "semantic_query": "semantic wording" if valid else None,
        "hypothetical_document": "synthetic passage" if valid else None,
        "subqueries": ["aspect one", "aspect two"] if valid else [],
        "usage": {
            "prompt_tokens": 10, "output_tokens": 5, "seconds": 0.5,
            "incurred_prompt_tokens": 10, "incurred_output_tokens": 5, "incurred_seconds": 0.5,
        },
    }


class SearchActionsTests(unittest.TestCase):
    def test_failed_semantic_generation_retains_task_and_generation_cost(self):
        search = SimpleNamespace(batch=Mock(side_effect=lambda queries: [["relevant"] for query in queries]))
        ranks, costs = runner.search_actions("how does compound interest grow savings", generated_row(valid=False), search)
        self.assertEqual(len(ranks), 5)
        self.assertEqual(ranks[2:], [[], [], []])
        self.assertEqual(len(search.batch.call_args.args[0]), 2, "No unavailable action should reach search")
        self.assertEqual([cost["search_calls"] for cost in costs], [1, 1, 0, 0, 0])
        self.assertEqual([cost["llm_calls"] for cost in costs], [0, 0, 1, 1, 1])
        self.assertEqual([cost["input_tokens"] for cost in costs], [0, 0, 10, 10, 10])
        self.assertEqual([cost["output_tokens"] for cost in costs], [0, 0, 5, 5, 5])
        self.assertEqual(runner.ndcg_at_k(ranks[2], {"relevant": 1}), 0.0)

    def test_decomposition_fuses_two_searches_and_charges_one_shared_generation(self):
        def batch(queries):
            ranks = {"aspect one": ["only-a", "shared"], "aspect two": ["only-b", "shared"]}
            return [ranks.get(query, ["baseline"]) for query in queries]

        search = SimpleNamespace(batch=Mock(side_effect=batch))
        ranks, costs = runner.search_actions("compound interest", generated_row(), search)
        self.assertEqual(ranks[-1], ["shared", "only-a", "only-b"])
        self.assertEqual([cost["search_calls"] for cost in costs], [1, 1, 1, 1, 2])
        self.assertEqual(costs[-1], {"search_calls": 2, "llm_calls": 1, "input_tokens": 10, "output_tokens": 5})
        self.assertEqual(len(search.batch.call_args.args[0]), 6)

    def test_virtual_cost_does_not_disappear_on_cached_generation(self):
        row = generated_row()
        row["cache"] = {"hit": True}
        for key in ("incurred_prompt_tokens", "incurred_output_tokens", "incurred_seconds"):
            row["usage"][key] = 0
        self.assertEqual(runner.generation_usage(row), {"llm_calls": 1, "input_tokens": 10, "output_tokens": 5, "seconds": 0.5})


class PreparationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.docs = [{"_id": doc_id, "title": "SECRET_SOURCE_TITLE", "text": "SECRET_SOURCE_BODY"} for doc_id in ("a", "b", "invalid")]
        self.ids = {"train": ["task-1"], "calibration": ["task-2"], "utility": ["task-3"], "test": ["task-4"]}
        self.queries = {f"task-{index}": f"real query {index}" for index in range(1, 5)}
        self.pools = {"11": ["a", "b", "invalid"], "23": ["b", "a", "invalid"]}
        self.qrels = {partition: {qid: {"SECRET_RELEVANCE_LABEL": 1} for qid in qids} for partition, qids in self.ids.items()}
        self.calls = []
        calls = self.calls

        class FakeGenerator:
            def __init__(self, model_path, cache_dir, batch_size):
                pass

            def probe_many(self, documents):
                calls.append(("probe", documents))
                return [generated_row(valid=document["_id"] != "invalid") | {
                    "direct_question": f"direct question {document['_id']}" if document["_id"] != "invalid" else None,
                    "indirect_question": f"indirect question {document['_id']}" if document["_id"] != "invalid" else None,
                } for document in documents]

            def rewrite_many(self, queries):
                calls.append(("rewrite", queries))
                if not all(isinstance(query, str) for query in queries):
                    raise AssertionError("Only strings may cross the query rewrite boundary")
                return [generated_row(valid=query != "real query 4") for query in queries]

        fake_module = ModuleType("generation")
        fake_module.FrozenGenerator = FakeGenerator
        self.args = SimpleNamespace(batch_size=8, test_queries=4, source_queries=4, probe_documents=3, phase="prepare")
        patches = [
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "DATASETS", ("fake-corpus",)),
            patch.object(runner, "dataset_inputs", return_value=(self.docs, self.queries, self.ids, self.pools, self.qrels, {"split": "fake"})),
            patch.object(runner, "code_contract", return_value={"frozen": "hash"}),
            patch.dict(sys.modules, {"generation": fake_module}),
        ]
        for active_patch in patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def test_document_to_question_then_query_only_rewrite_boundary(self):
        with redirect_stdout(io.StringIO()):
            runner.prepare(self.args)
        self.assertEqual([kind for kind, _ in self.calls], ["probe", "rewrite", "rewrite"])
        self.assertEqual([doc["_id"] for doc in self.calls[0][1]], ["a", "b", "invalid"], "Repeated pools share one generation per source")
        self.assertEqual(self.calls[1][1], ["direct question a", "indirect question a", "direct question b", "indirect question b"])
        self.assertEqual(self.calls[2][1], list(self.queries.values()))
        serialized_rewriter_input = json.dumps(self.calls[1:])
        self.assertNotIn("SECRET_SOURCE", serialized_rewriter_input)
        self.assertNotIn("SECRET_RELEVANCE_LABEL", serialized_rewriter_input)
        prepared = json.loads((self.root / "cache" / "prepared.json").read_text())
        data = prepared["datasets"]["fake-corpus"]
        self.assertEqual(data["ids"]["test"], ["task-4"])
        self.assertFalse(data["task_rewrites"]["task-4"]["valid"], "A failed rewrite must not remove its evaluation task")
        self.assertEqual(len(data["probe_rewrites"]), 4)
        audit = json.loads((self.root / "results" / "generation_audit.json").read_text())["datasets"]["fake-corpus"]
        self.assertEqual(audit["task_count"], 4)
        self.assertEqual(audit["valid_task_rewrites"], 3)
        self.assertEqual(audit["logical_generation_cost"]["llm_calls"], 11)
        self.assertEqual(audit["actual_generation_cost"]["output_tokens"], 55)

    def test_generation_contract_drift_refuses_to_publish_prepared_file(self):
        with patch.object(runner, "code_contract", side_effect=[{"frozen": "before"}, {"frozen": "after"}]), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                runner.prepare(self.args)
        self.assertFalse((self.root / "cache" / "prepared.json").exists())

    def test_retrieval_rejects_manifest_drift_before_loading_models(self):
        (self.root / "cache").mkdir()
        (self.root / "cache" / "prepared.json").write_text(json.dumps({"contract": {"different": "hash"}}))
        with patch.object(runner, "DenseRetriever", side_effect=AssertionError("model must not load")):
            with self.assertRaisesRegex(ValueError, "different contract"):
                runner.retrieve(self.args)


class DownloaderIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_existing_archive_checks_bytes_and_never_downloads(self):
        path = self.root / "nfcorpus.zip"
        path.write_bytes(b"test fixture archive")
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(downloader, "NFCORPUS_SHA256", expected), patch.object(downloader.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            result = downloader.ensure_nfcorpus(path)
        self.assertEqual(result["sha256"], expected)
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            downloader.verify_archive(path, expected)

    def test_verify_only_missing_archive_does_not_download(self):
        with patch.object(downloader.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(FileNotFoundError):
                downloader.ensure_nfcorpus(self.root / "missing.zip", verify_only=True)

    def test_corrupt_download_never_replaces_final_archive(self):
        path = self.root / "nfcorpus.zip"
        with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt download")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                downloader.ensure_nfcorpus(path)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob("*.download")), [])

    def test_pinned_snapshot_requires_tokenizer_template_and_every_weight_shard(self):
        path = downloader.snapshot_path(self.root, downloader.QWEN_REPO, downloader.QWEN_REVISION)
        path.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "first.safetensors"):
            (path / name).write_text("fixture")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}))
        with self.assertRaisesRegex(FileNotFoundError, "second.safetensors"):
            downloader.verify_snapshot(path, downloader.QWEN_REPO, downloader.QWEN_REVISION)
        (path / "second.safetensors").write_text("fixture")
        verified = downloader.verify_snapshot(path, downloader.QWEN_REPO, downloader.QWEN_REVISION)
        self.assertEqual(verified["revision"], downloader.QWEN_REVISION)
        with self.assertRaisesRegex(ValueError, "immutable snapshot"):
            downloader.verify_snapshot(path, downloader.QWEN_REPO, "0" * 40)


if __name__ == "__main__":
    unittest.main()
