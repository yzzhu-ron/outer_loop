"""Offline generation contracts; these tests are not semantic quality evidence."""

import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("semantic_generation_test", Path(__file__).resolve().parents[1] / "generation.py")
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)


def valid_rewrite():
    return {
        "semantic_query": "How does compound interest change long term savings?",
        "hypothetical_document": " ".join(["synthetic"] * 45),
        "subqueries": ["Compound interest and long term savings growth", "How reinvesting interest changes investment returns"],
    }


def valid_probe():
    return {
        "direct_question": "How does compound interest affect savings over time?",
        "indirect_question": "Why can reinvesting accumulated returns accelerate the growth of money?",
    }


class FakeBackend:
    calls = []
    raw_response = None
    finish_reason = "stop"

    def __init__(self, model_path):
        self.model_path = model_path

    def generate(self, messages, max_tokens):
        type(self).calls.append((messages, max_tokens))
        return [{
            "text": type(self).raw_response if type(self).raw_response is not None else json.dumps(
                valid_rewrite() if "query" in json.loads(conversation[-1]["content"]) else valid_probe()
            ),
            "prompt_tokens": 123,
            "output_tokens": 80,
            "output_tokens_without_stop": 79,
            "finish_reason": type(self).finish_reason,
            "batch_id": f"batch-{len(type(self).calls)}",
            "batch_size": len(messages),
            "batch_wall_seconds": 2.0,
            "batch_prompt_seconds": 0.5,
            "batch_decode_seconds": 1.5,
            "batch_prompt_tokens": 123 * len(messages),
            "batch_output_tokens": 80 * len(messages),
            "peak_memory_gb": 1.0,
        } for conversation in messages]


class ParseTests(unittest.TestCase):
    def test_valid_output_retains_synthetic_and_unverified_semantic_status(self):
        actual = generation.parse_response(json.dumps(valid_rewrite()), "rewrite")
        self.assertTrue(actual["valid"])
        self.assertTrue(actual["hypothetical_document_is_synthetic"])
        self.assertEqual(actual["validity"]["semantic"], "not_automatically_verified")
        self.assertIn("semantic_faithfulness_requires_human_audit", actual["warnings"])

    def test_strict_json_no_fence_extraction_duplicates_nonfinite_or_extra_text(self):
        raw = json.dumps(valid_rewrite())
        for invalid in ("", " ", "```json\n" + raw + "\n```", "Here: " + raw, raw + " trailing", "null", "[]", '{"semantic_query":"one","semantic_query":"two"}', '{"x":NaN}'):
            with self.subTest(raw=invalid):
                actual = generation.parse_response(invalid, "rewrite")
                self.assertFalse(actual["valid"])
                self.assertTrue(actual["errors"])
                self.assertIsNone(actual["semantic_query"])
                self.assertEqual(actual["subqueries"], [])

    def test_nonempty_exact_schema_and_bounded_passage(self):
        for changes in ({"semantic_query": " "}, {"subqueries": ["one"]}, {"subqueries": ["one", 2]}, {"extra": "field"}, {"hypothetical_document": "short"}, {"hypothetical_document": " ".join(["word"] * 19)}, {"hypothetical_document": " ".join(["word"] * 101)}, {"semantic_query": " ".join(["word"] * 61)}):
            with self.subTest(changes=changes):
                payload = valid_rewrite() | changes
                actual = generation.parse_response(json.dumps(payload), "rewrite")
                self.assertFalse(actual["valid"])
                self.assertIsNone(actual["hypothetical_document"])
        for count in (40, 60):
            payload = valid_rewrite() | {"hypothetical_document": " ".join(["word"] * count)}
            self.assertTrue(generation.parse_response(json.dumps(payload), "rewrite")["valid"])

    def test_passage_soft_word_budget_retains_all_other_rewrite_fields(self):
        for count in (20, 35, 39, 61, 100):
            payload = valid_rewrite() | {"hypothetical_document": " ".join(["word"] * count)}
            actual = generation.parse_response(json.dumps(payload), "rewrite")
            self.assertTrue(actual["valid"])
            self.assertEqual(actual["errors"], [])
            self.assertEqual(actual["semantic_query"], payload["semantic_query"])
            self.assertEqual(actual["subqueries"], payload["subqueries"])
            self.assertIn(f"hypothetical_document_outside_requested_40_60_words: {count}", actual["warnings"])

    def test_probe_copying_is_an_audit_warning_not_semantic_validation(self):
        source = {"title": "The source document title", "text": "one two three four five six seven eight"}
        payload = {"direct_question": "How?", "indirect_question": "The source document title one two three four five six?"}
        actual = generation.parse_response(json.dumps(payload), "probe", source)
        self.assertTrue(actual["valid"])
        self.assertIn("indirect_question_contains_source_title", actual["warnings"])
        self.assertIn("indirect_question_copies_source_span_of_at_least_six_words", actual["warnings"])
        self.assertEqual(actual["validity"]["semantic"], "not_automatically_verified")

    def test_same_questions_and_subqueries_are_flagged(self):
        payload = valid_probe() | {"indirect_question": valid_probe()["direct_question"]}
        self.assertIn("identical_direct_and_indirect_questions", generation.parse_response(json.dumps(payload), "probe")["warnings"])
        payload = valid_rewrite() | {"subqueries": ["same", "same"]}
        self.assertIn("identical_subqueries_require_audit", generation.parse_response(json.dumps(payload), "rewrite")["warnings"])


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "models--fake" / "snapshots" / ("a" * 40)
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text('{"test": true}')
        self.cache = self.root / "cache"
        FakeBackend.calls = []
        FakeBackend.raw_response = None
        FakeBackend.finish_reason = "stop"
        self.backend_patch = patch.object(generation, "_MLXBackend", FakeBackend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    def make_generator(self, **kwargs):
        return generation.FrozenGenerator(self.model, self.cache, **kwargs)

    def test_batched_order_and_exactly_once_cost_for_duplicates(self):
        generator = self.make_generator(batch_size=2)
        output = generator.rewrite_many(["first query", "second query", "first query", "third query"])
        self.assertEqual([len(call[0]) for call in FakeBackend.calls], [2, 1])
        self.assertEqual([row["cache"]["hit"] for row in output], [False, False, True, False])
        self.assertEqual(output[0]["cache"]["key"], output[2]["cache"]["key"])
        self.assertEqual(sum(row["usage"]["incurred_prompt_tokens"] for row in output), 369)
        self.assertEqual(sum(row["usage"]["incurred_output_tokens"] for row in output), 240)
        self.assertEqual(sum(row["usage"]["incurred_seconds"] for row in output), 4.0)
        self.assertEqual(output[0]["usage"]["output_tokens_without_stop"], 79)
        self.assertEqual(output[0]["usage"]["seconds"], 1.0)
        self.assertIn("divided equally", output[0]["usage"]["time_accounting"])

    def test_disk_cache_does_not_load_model_or_reincur_cost(self):
        original = self.make_generator().rewrite_many(["savings question"])[0]
        with patch.object(generation, "_MLXBackend", side_effect=AssertionError("cache should avoid model load")):
            cached = self.make_generator().rewrite_many(["savings question"])[0]
        self.assertEqual(cached["raw_response"], original["raw_response"])
        self.assertEqual(cached["usage"]["prompt_tokens"], original["usage"]["prompt_tokens"])
        self.assertTrue(cached["cache"]["hit"])
        self.assertEqual(cached["usage"]["incurred_prompt_tokens"], 0)
        self.assertEqual(cached["usage"]["incurred_output_tokens"], 0)
        self.assertEqual(cached["usage"]["incurred_seconds"], 0.0)

    def test_full_input_model_prompt_parameters_and_revision_affect_cache_key(self):
        first = self.make_generator()
        baseline = generation._digest(first.cache_identity("rewrite", "query"))
        alternatives = [first.cache_identity("rewrite", "query changed"), self.make_generator(batch_size=4).cache_identity("rewrite", "query")]
        with patch.object(generation, "PROMPT_VERSION", "different-version"):
            alternatives.append(first.cache_identity("rewrite", "query"))
        with patch.object(generation, "_REWRITE_SYSTEM", "different text same version"):
            alternatives.append(first.cache_identity("rewrite", "query"))
        second_model = self.model.parent / ("b" * 40)
        second_model.mkdir()
        second = generation.FrozenGenerator(second_model, self.cache)
        alternatives.append(second.cache_identity("rewrite", "query"))
        (self.model / "config.json").write_text('{"test": false}')
        alternatives.append(self.make_generator().cache_identity("rewrite", "query"))
        self.assertTrue(all(generation._digest(identity) != baseline for identity in alternatives))

    def test_probe_id_and_entire_untruncated_input_in_cache_not_prompt(self):
        generator = self.make_generator()
        source = {"_id": "DOC-ID-SECRET", "title": "Sample title", "text": "text " * 3000, "relevance_label": 99}
        changed = source | {"text": source["text"] + " a changed suffix"}
        results = generator.probe_many([source, changed])
        self.assertNotEqual(results[0]["cache"]["key"], results[1]["cache"]["key"])
        messages = FakeBackend.calls[0][0]
        self.assertEqual(messages[0], messages[1], "Only explicitly bounded title and text reach the model")
        content = json.loads(messages[0][-1]["content"])
        self.assertEqual(set(content), {"title", "text"})
        self.assertNotIn("DOC-ID-SECRET", json.dumps(messages))
        self.assertNotIn("relevance_label", json.dumps(messages))
        self.assertTrue(any("source_truncated" in warning for warning in results[0]["warnings"]))

    def test_source_never_enters_query_rewrite_in_second_stage(self):
        generator = self.make_generator()
        document = {"_id": "ID-MARKER", "title": "TITLE-MARKER", "text": "SOURCE-CONTENT-MARKER describes something specific."}
        probe = generator.probe_many([document])[0]
        query = probe["indirect_question"]
        generator.rewrite_many([query])
        messages = FakeBackend.calls[-1][0][0]
        self.assertEqual(json.loads(messages[-1]["content"]), {"query": query})
        self.assertNotIn("SOURCE-CONTENT-MARKER", json.dumps(messages))
        self.assertNotIn("TITLE-MARKER", json.dumps(messages))
        self.assertNotIn("ID-MARKER", json.dumps(messages))
        real_query_generator = generation.FrozenGenerator(self.model, self.root / "other-cache")
        real_query_generator.rewrite_many([query])
        self.assertEqual(FakeBackend.calls[-1][0][0], messages)

    def test_invalid_output_cached_logged_and_never_falls_back_or_retries(self):
        FakeBackend.raw_response = "not json"
        first = self.make_generator().rewrite_many(["original lexical query"])[0]
        self.assertFalse(first["valid"])
        self.assertIsNone(first["semantic_query"])
        self.assertEqual(first["raw_response"], "not json")
        self.assertEqual(len(FakeBackend.calls), 1)
        FakeBackend.raw_response = None
        second = self.make_generator().rewrite_many(["original lexical query"])[0]
        self.assertFalse(second["valid"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(len(FakeBackend.calls), 1)
        logged = (self.cache / "invalid_generations.jsonl").read_text().splitlines()
        self.assertEqual(len(logged), 1)
        self.assertEqual(json.loads(logged[0])["record"]["raw_response"], "not json")

    def test_length_truncation_invalidates_even_an_apparently_complete_object(self):
        FakeBackend.finish_reason = "length"
        output = self.make_generator().rewrite_many(["query"])[0]
        self.assertFalse(output["valid"])
        self.assertIsNone(output["semantic_query"])
        self.assertIn("generation_finish_reason: length", output["errors"])

    def test_invalid_inputs_rejected_before_model_load(self):
        generator = self.make_generator()
        for queries in ([""], [" "], [None], ["x" * (generation.MAX_QUERY_CHARS + 1)]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                generator.rewrite_many(queries)
        for docs in ([{}], [{"text": ""}], [{"text": "body", "_id": 5}], ["wrong type"]):
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                generator.probe_many(docs)
        self.assertEqual(generator.rewrite_many([]), [])
        self.assertEqual(generator.probe_many([]), [])
        self.assertEqual(FakeBackend.calls, [])
        for batch_size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.make_generator(batch_size=batch_size)

    def test_cache_identity_mismatch_fails_explicitly(self):
        generator = self.make_generator()
        result = generator.rewrite_many(["query"])[0]
        path = self.cache / (result["cache"]["key"] + ".json")
        cached = json.loads(path.read_text())
        cached["identity"]["input"] = "different query"
        path.write_text(json.dumps(cached))
        with self.assertRaisesRegex(ValueError, "Cache identity mismatch"):
            generator.rewrite_many(["query"])

    def test_model_revision_mismatch_and_copied_weight_change(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.make_generator(model_revision="c" * 40)
        local = self.root / "local-model"
        local.mkdir()
        (local / "model.safetensors").write_bytes(b"first fake weights")
        first = generation._model_identity(local, "label")
        (local / "model.safetensors").write_bytes(b"other fake weights")
        self.assertNotEqual(first, generation._model_identity(local, "label"))


class BackendAccountingTests(unittest.TestCase):
    def test_real_adapter_counts_sampled_tokens_instead_of_retokenizing_text(self):
        """Exercise MLX adapter control flow with a fake token stream, offline."""
        mx = ModuleType("mlx.core")
        mlx = ModuleType("mlx")
        mlx.core = mx
        lm = ModuleType("mlx_lm")
        lm_generate = ModuleType("mlx_lm.generate")
        sampler = ModuleType("mlx_lm.sample_utils")
        observed = {}

        class Tokenizer:
            eos_token_ids = [99]

            def apply_chat_template(self, conversation, **kwargs):
                observed["template_kwargs"] = kwargs
                return "chat template rendered"

            def encode(self, rendered, **kwargs):
                observed["encode_kwargs"] = kwargs
                return [10, 11, 12, 13]

            def decode(self, tokens):
                return "decoded:" + ",".join(map(str, tokens))

        class BatchGenerator:
            def __init__(self, model, **kwargs):
                observed["generator_kwargs"] = kwargs
                self.step = 0

            def insert(self, prompts, max_tokens):
                observed["insert"] = (prompts, max_tokens)
                return [40, 41]

            @contextlib.contextmanager
            def stats(self):
                yield SimpleNamespace(prompt_time=0.1, generation_time=0.2, peak_memory=3.0)

            def next_generated(self):
                self.step += 1
                sequences = [
                    [SimpleNamespace(uid=40, token=7, finish_reason=None), SimpleNamespace(uid=41, token=8, finish_reason=None)],
                    [SimpleNamespace(uid=40, token=99, finish_reason="stop"), SimpleNamespace(uid=41, token=9, finish_reason="length")],
                    [],
                ]
                return sequences[self.step - 1]

            def close(self):
                observed["closed"] = True

        lm.load = lambda *args, **kwargs: (SimpleNamespace(eval=lambda: None), Tokenizer())
        lm_generate.BatchGenerator = BatchGenerator
        sampler.make_sampler = lambda **kwargs: "greedy_sampler"
        modules = {"mlx": mlx, "mlx.core": mx, "mlx_lm": lm, "mlx_lm.generate": lm_generate, "mlx_lm.sample_utils": sampler}
        with patch.dict(sys.modules, modules):
            adapter = generation._MLXBackend(Path("unused-fake-model"))
            result = adapter.generate([[{"role": "user", "content": "first"}], [{"role": "user", "content": "second"}]], 2)
        self.assertEqual([row["prompt_tokens"] for row in result], [4, 4])
        self.assertEqual([row["output_tokens"] for row in result], [2, 2])
        self.assertEqual([row["output_tokens_without_stop"] for row in result], [1, 2])
        self.assertEqual([row["finish_reason"] for row in result], ["stop", "length"])
        self.assertEqual([row["text"] for row in result], ["decoded:7", "decoded:8,9"])
        self.assertEqual(result[0]["batch_id"], result[1]["batch_id"])
        self.assertEqual(result[0]["batch_prompt_tokens"], 8)
        self.assertEqual(result[0]["batch_output_tokens"], 4)
        self.assertEqual(observed["encode_kwargs"], {"add_special_tokens": False})
        self.assertEqual(observed["generator_kwargs"]["stop_tokens"], [[99]])
        self.assertEqual(observed["generator_kwargs"]["sampler"], "greedy_sampler")
        self.assertTrue(observed["closed"])


if __name__ == "__main__":
    unittest.main()
