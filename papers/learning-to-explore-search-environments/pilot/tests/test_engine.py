"""Offline retrieval correctness and information-contract checks.

The tiny fake dense encoder exercises caching, ranking and empty-query behavior;
it is not evidence for the quality of the real sentence-transformer model.
"""

from collections import Counter
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


_ENGINE_PATH = Path(__file__).resolve().parents[1] / "engine.py"
_SPEC = importlib.util.spec_from_file_location("pilot_engine_for_test", _ENGINE_PATH)
engine = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(engine)


class LexicalTests(unittest.TestCase):
    def test_query_only_actions_preserve_identifiers(self):
        query = "How can we compare signal activity in several different cell types; specifically IL-6, p53 and C++ foo_bar v1.2?"
        expected = {"il-6", "p53", "c++", "foo_bar", "v1.2"}
        for action in engine.ACTIONS:
            with self.subTest(action=action):
                first = engine.transform(query, action, {"signal": 900})
                second = engine.transform(query, action, {"cell": 100000})
                self.assertEqual(first, second, "Corpus IDF must not influence query transformations")
                self.assertTrue(expected <= set(engine.tokenize(first)))
        self.assertEqual(engine.transform(query, "original"), query)
        self.assertEqual(engine.transform("how can the", "keywords"), "how can the")
        self.assertEqual(engine.transform("?!", "keywords"), "")
        with self.assertRaises(ValueError):
            engine.transform(query, "unregistered")

    def test_features_use_fixed_buckets_and_no_nan_for_empty_text(self):
        self.assertEqual(engine.feature_bucket("cell activity"), 0)
        self.assertEqual(engine.feature_bucket("IL-6 activity"), 1)
        self.assertEqual(engine.feature_bucket(" ".join(["cell"] * 8)), 0)
        self.assertEqual(engine.feature_bucket(" ".join(["cell"] * 9)), 2)
        self.assertEqual(engine.feature_bucket(" ".join(["cell"] * 8 + ["p53"])), 3)
        features = engine.query_features("")
        self.assertEqual(len(features), len(engine.FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in features))
        self.assertEqual(features, [0.0] * len(features))

    def test_body_probe_is_title_excluded_spaced_and_document_only(self):
        document = {
            "_id": "not-an-input-to-query-generation",
            "title": "Immune receptor IL-6",
            "text": "Immune receptor IL-6 " + " ".join(f"term{index}" for index in range(50)),
        }
        probe = engine.generate_probe(document, "body_terms")
        tokens = engine.tokenize(probe)
        self.assertEqual(len(tokens), 12)
        self.assertFalse(set(tokens) & set(engine.tokenize(document["title"])))
        self.assertEqual(tokens[0], "term0")
        self.assertEqual(tokens[-1], "term48")
        changed_id = dict(document, _id="completely-different")
        self.assertEqual(probe, engine.generate_probe(changed_id, "body_terms"))
        source = engine.tokenize(document["text"])
        source_trigrams = {tuple(source[index:index + 3]) for index in range(len(source) - 2)}
        self.assertFalse(any(tuple(tokens[index:index + 3]) in source_trigrams for index in range(len(tokens) - 2)))
        self.assertEqual(engine.generate_probe(document, "exact_title"), document["title"])
        self.assertEqual(engine.generate_probe({"title": "", "text": "the and of"}, "body_terms"), "")
        self.assertEqual(engine.generate_probe({"title": "?!"}, "exact_title"), "")


class BM25Tests(unittest.TestCase):
    def test_scores_match_independent_formula_and_keep_all_corpus_documents(self):
        documents = [
            {"_id": "z", "title": "", "text": "apple apple banana"},
            {"_id": "a", "title": "apple", "text": "banana banana pear"},
            {"_id": "long", "title": "", "text": "pear " * 10 + "apple"},
            {"_id": "empty", "title": "", "text": ""},
        ]
        retriever = engine.BM25(documents)
        bags = [Counter(engine.tokenize(document["title"] + " " + document["text"])) for document in documents]
        average_length = sum(sum(bag.values()) for bag in bags) / len(bags)
        query = Counter(["apple", "apple", "banana"])
        expected_scores = []
        for bag in bags:
            score = 0.0
            for term, query_count in query.items():
                frequency = bag[term]
                document_frequency = sum(term in other for other in bags)
                idf = math.log(1 + (len(bags) - document_frequency + 0.5) / (document_frequency + 0.5))
                score += query_count * idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * sum(bag.values()) / average_length))
            expected_scores.append(score)
        indices = [retriever._vocabulary[term] for term in query]
        actual_scores = retriever._weights[:, indices] @ np.array(list(query.values()))
        np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-12)
        expected_order = sorted(range(len(bags) - 1), key=lambda index: (-expected_scores[index], index))
        self.assertEqual(retriever.search("apple apple banana"), [documents[index]["_id"] for index in expected_order])
        self.assertEqual(retriever.document_ids, [document["_id"] for document in documents])

    def test_empty_oov_ties_and_batch_interface(self):
        retriever = engine.BM25([
            {"_id": "z", "text": "same term"},
            {"_id": "a", "text": "same term"},
            {"_id": "unmatched", "text": "other"},
        ])
        self.assertEqual(retriever.search("same", k=1), ["z"])
        self.assertEqual(retriever.search("same", k=10), ["z", "a"])
        self.assertEqual(retriever.batch_search(["", "?!", "absent", "same"], k=1), [[], [], [], ["z"]])
        self.assertEqual(retriever.search("same", k=0), [])
        self.assertEqual(engine.BM25([]).search("anything"), [])
        with self.assertRaises(ValueError):
            engine.BM25([{"_id": "same"}, {"_id": "same"}])


class _FakeSentenceTransformer:
    def __init__(self, model_name, revision, device, trust_remote_code):
        self.tokenizer = SimpleNamespace(model_max_length=512)
        self._first = SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)))
        self.calls = []

    def __getitem__(self, index):
        return self._first

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([-1.0, 0.0, 0.0])
        return np.array(vectors, dtype=np.float32)


class DenseTests(unittest.TestCase):
    def setUp(self):
        torch = ModuleType("torch")
        torch.set_num_threads = lambda count: None
        sentence_transformers = ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
        self.module_patch = patch.dict(sys.modules, {"torch": torch, "sentence_transformers": sentence_transformers})
        self.version_patch = patch.object(engine.importlib.metadata, "version", return_value="offline-test")
        self.module_patch.start()
        self.version_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self.version_patch.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.documents = [
            {"_id": "z", "text": "alpha"},
            {"_id": "a", "text": "alpha"},
            {"_id": "orthogonal", "text": "beta"},
            {"_id": "negative", "text": "gamma"},
        ]

    def make(self, documents=None, **kwargs):
        return engine.DenseRetriever(
            self.documents if documents is None else documents,
            self.temporary.name, revision=kwargs.pop("revision", "a" * 40),
            batch_size=2, **kwargs,
        )

    def test_dense_ranking_boundary_ties_negative_scores_and_empty_queries(self):
        retriever = self.make()
        self.assertEqual(retriever.search("alpha", 1), ["z"])
        self.assertEqual(retriever.search("alpha", 4), ["z", "a", "orthogonal", "negative"])
        before = len(retriever.model.calls)
        self.assertEqual(retriever.batch_search(["", "?!"]), [[], []])
        self.assertEqual(len(retriever.model.calls), before, "Empty queries must never reach the encoder")
        self.assertEqual(retriever.batch_search(["", "alpha", "beta"], 1), [[], ["z"], ["orthogonal"]])

    def test_cache_depends_on_full_corpus_order_revision_and_truncation(self):
        first = self.make()
        reused = self.make()
        self.assertEqual(first.cache_key, reused.cache_key)
        self.assertEqual(reused.model.calls, [], "A valid cache should avoid document encoding")
        changed_body = [dict(document) for document in self.documents]
        changed_body[0]["text"] += " extra" * 1000
        variants = [
            self.make(documents=changed_body),
            self.make(documents=list(reversed(self.documents))),
            self.make(revision="b" * 40),
            self.make(max_seq_length=128),
        ]
        self.assertEqual(len({first.cache_key, *(variant.cache_key for variant in variants)}), 5)
        with self.assertRaises(ValueError):
            self.make(revision="main")
        with self.assertRaises(ValueError):
            self.make(max_seq_length=513)


class MetricTests(unittest.TestCase):
    def test_graded_ndcg_recall_and_known_document_reciprocal_rank(self):
        qrels = {"strong": 2, "weak": 1, "irrelevant": 0}
        self.assertAlmostEqual(engine.ndcg_at_k(["strong", "weak"], qrels), 1.0)
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(engine.ndcg_at_k(["weak", "strong"], qrels), expected)
        self.assertEqual(engine.ndcg_at_k(["missing"], qrels), 0.0)
        self.assertEqual(engine.ndcg_at_k(["strong"], {}), 0.0)
        self.assertLess(engine.ndcg_at_k(["strong", "strong"], qrels), 1.0)
        self.assertEqual(engine.recall_at_k(["strong", "strong"], qrels), 0.5)
        self.assertEqual(engine.recall_at_k(["weak", "strong"], qrels), 1.0)
        self.assertEqual(engine.reciprocal_rank(["missing", "strong"], "strong"), 0.5)
        self.assertEqual(engine.reciprocal_rank(["strong"], "weak"), 0.0)


if __name__ == "__main__":
    unittest.main()
