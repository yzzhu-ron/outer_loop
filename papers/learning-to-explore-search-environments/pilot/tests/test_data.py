"""Checks for the pilot's data and evaluation information boundary."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import eligible_queries, load_dataset, make_split


class SplitTests(unittest.TestCase):
    def test_duplicate_bodies_stay_together_despite_title_variants(self):
        docs = [
            {"_id": "a", "title": "First", "text": "Ａlpha\n BETA"},
            {"_id": "b", "title": "Other", "text": "alpha beta"},
            {"_id": "c", "title": " Empty body ", "text": ""},
            {"_id": "d", "title": "empty   BODY", "text": "  "},
            {"_id": "e", "title": "Unique", "text": "different content"},
        ]
        for seed in range(20):
            exploration, heldout, audit = make_split(docs, seed=seed, exploration_fraction=0.5)
            self.assertEqual("a" in exploration, "b" in exploration)
            self.assertEqual("c" in exploration, "d" in exploration)
            self.assertFalse(exploration & heldout)
            self.assertEqual(exploration | heldout, {"a", "b", "c", "d", "e"})
            self.assertEqual(audit["duplicate_group_count"], 2)
            self.assertEqual(audit["duplicate_excess_document_count"], 2)

    def test_deterministic_and_order_independent(self):
        docs = [{"_id": str(i), "title": "", "text": f"document {i}"} for i in range(200)]
        first = make_split(docs)
        self.assertEqual(first, make_split(list(reversed(docs))))
        self.assertEqual(first, make_split(docs))
        self.assertNotEqual(first[0], make_split(docs, seed=20260910)[0])
        self.assertEqual(len(docs), 200)

    def test_fraction_boundaries_and_invalid_split(self):
        docs = [{"_id": "d", "text": "body", "title": ""}]
        self.assertEqual(make_split(docs, exploration_fraction=0)[0], set())
        self.assertEqual(make_split(docs, exploration_fraction=1)[0], {"d"})
        for fraction in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                make_split(docs, exploration_fraction=fraction)
        with self.assertRaises(ValueError):
            make_split(docs + docs)

    def test_all_positive_support_must_be_held_out(self):
        qrels = {
            "keep": {"h1": 2, "h2": 1, "e": 0},
            "mixed": {"h1": 1, "e": 1},
            "exploration_only": {"e": 1},
            "no_positive": {"h1": 0},
        }
        kept, audit = eligible_queries(qrels, {"h1", "h2"}, {"h1", "h2", "e"})
        self.assertEqual(kept, {"keep": qrels["keep"]})
        self.assertEqual(audit["partially_heldout_query_count"], 1)
        self.assertEqual(audit["no_heldout_support_query_count"], 1)
        self.assertEqual(audit["no_positive_support_query_count"], 1)
        self.assertEqual(audit["query_retention_fraction"], 0.25)
        kept["keep"]["h1"] = 99
        self.assertEqual(qrels["keep"]["h1"], 2)

    def test_missing_judged_documents_are_excluded_and_audited(self):
        qrels = {
            "positive_missing": {"heldout": 1, "missing": 1},
            "zero_missing": {"heldout": 1, "other_missing": 0},
            "keep": {"heldout": 1},
        }
        kept, audit = eligible_queries(qrels, {"heldout"}, {"heldout"})
        self.assertEqual(set(kept), {"keep"})
        self.assertEqual(audit["missing_document_query_count"], 2)
        self.assertEqual(audit["missing_document_ids"], ["missing", "other_missing"])
        with self.assertRaises(ValueError):
            eligible_queries(qrels, {"not_in_corpus"}, {"heldout"})


class ArchiveTests(unittest.TestCase):
    def _write_archive(self, path, *, corpus=None, qrels=None):
        corpus = corpus or [{"_id": "d1", "title": "Title", "text": "Body", "metadata": {}}]
        with ZipFile(path, "w") as archive:
            archive.writestr("tiny/corpus.jsonl", "\n".join(json.dumps(row) for row in corpus))
            archive.writestr("tiny/queries.jsonl", json.dumps({"_id": "q1", "text": "Question"}))
            archive.writestr("tiny/qrels/test.tsv", qrels or "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
            archive.writestr("../../must-not-extract.txt", "not extracted")

    def test_read_streams_without_extracting_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            self._write_archive(data_dir / "tiny.zip")
            docs, queries, qrels = load_dataset(data_dir, "tiny")
            self.assertEqual(docs, [{"_id": "d1", "title": "Title", "text": "Body"}])
            self.assertEqual(queries, {"q1": "Question"})
            self.assertEqual(qrels, {"test": {"q1": {"d1": 1}}})
            self.assertEqual([p.name for p in data_dir.iterdir()], ["tiny.zip"])
            with self.assertRaises(ValueError):
                load_dataset(data_dir, "../tiny")

    def test_reject_duplicate_document_ids_and_unknown_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            duplicate = {"_id": "d1", "title": "", "text": "text"}
            self._write_archive(data_dir / "tiny.zip", corpus=[duplicate, duplicate])
            with self.assertRaisesRegex(ValueError, "duplicate document ID"):
                load_dataset(data_dir, "tiny")
            self._write_archive(data_dir / "tiny.zip", qrels="query-id\tcorpus-id\tscore\nunknown\td1\t1\n")
            with self.assertRaisesRegex(ValueError, "unknown query"):
                load_dataset(data_dir, "tiny")

    def test_missing_document_reference_is_preserved_for_eligibility_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            self._write_archive(data_dir / "tiny.zip", qrels="query-id\tcorpus-id\tscore\nq1\tmissing\t1\n")
            docs, _, qrels = load_dataset(data_dir, "tiny")
            kept, audit = eligible_queries(qrels["test"], {"d1"}, {d["_id"] for d in docs})
            self.assertEqual(kept, {})
            self.assertEqual(audit["missing_document_count"], 1)


if __name__ == "__main__":
    unittest.main()
