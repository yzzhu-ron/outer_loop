"""Synthetic ranking mechanics only; no relevance labels or cached utilities."""
import math
import unittest

from adapter import (
    FrozenHybridBackend,
    action_queries,
    fuse_subqueries,
    weighted_rrf,
)


class WeightedRankingTests(unittest.TestCase):
    def test_endpoints_preserve_short_and_empty_support(self):
        # Zero-weight documents must never pad a short or empty active list.
        bm25, dense = ["z", "a"], ["other", "z", "tail"]
        self.assertEqual(weighted_rrf(bm25, dense, 0), bm25)
        self.assertEqual(weighted_rrf(bm25, dense, 1), dense)
        self.assertEqual(weighted_rrf([], dense, 0), [])
        self.assertEqual(weighted_rrf(bm25, [], 1), [])
        self.assertEqual(weighted_rrf([], dense, 0.25), dense)
        self.assertEqual(weighted_rrf(bm25, [], 0.75), bm25)
        self.assertEqual(weighted_rrf([], [], 0.5), [])

    def test_common_documents_accumulate_and_score_ties_use_document_id(self):
        # The common rank-two document receives twice the single-list weight.
        self.assertEqual(
            weighted_rrf(["z", "shared"], ["a", "shared"], 0.5),
            ["shared", "a", "z"],
        )
        self.assertEqual(weighted_rrf(["z"], ["a"], 0.5), ["a", "z"])
        self.assertEqual(weighted_rrf(["a"], ["z"], 0.5), ["a", "z"])

    def test_cutoff_and_dominant_endpoint_support(self):
        bm25 = [f"b{i:02}" for i in range(10)]
        dense = [f"d{i:02}" for i in range(10)]
        # At these weights every active-side top-ten score dominates every
        # other-side score: .75/70 > .25/61 for RRF's constant sixty.
        self.assertEqual(weighted_rrf(bm25, dense, 0.25), bm25)
        self.assertEqual(weighted_rrf(bm25, dense, 0.75), dense)
        middle = weighted_rrf(bm25, dense, 0.5)
        self.assertEqual(len(middle), 10)
        self.assertEqual(len(set(middle)), 10)
        self.assertEqual(
            middle, [item for pair in zip(bm25[:5], dense[:5]) for item in pair]
        )

    def test_invalid_weights_and_rank_lists_are_rejected(self):
        for weight in (-0.01, 1.01, math.nan, math.inf, -math.inf):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                weighted_rrf(["a"], ["b"], weight)
        for malformed in (["a", "a"], [str(i) for i in range(11)]):
            with self.subTest(ranking=malformed):
                with self.assertRaises(ValueError):
                    weighted_rrf(malformed, ["b"], 0.5)
                with self.assertRaises(ValueError):
                    weighted_rrf(["b"], malformed, 0.5)

    def test_unweighted_subquery_fusion_handles_empty_and_shared_results(self):
        self.assertEqual(fuse_subqueries([]), [])
        self.assertEqual(fuse_subqueries([[], []]), [])
        self.assertEqual(fuse_subqueries([["z", "a"]]), ["z", "a"])
        self.assertEqual(
            fuse_subqueries([["z", "shared"], ["a", "shared"]]),
            ["shared", "a", "z"],
        )


class FrozenBackendTests(unittest.TestCase):
    def setUp(self):
        self.bm25 = {"q1": ["a", "b"], "q2": ["a", "b"], "empty": []}
        self.dense = {"q1": ["b", "a"], "q2": ["c", "b"], "empty": []}

    def test_decomposed_hybrid_fuses_raw_subqueries_before_action_fusion(self):
        backend = FrozenHybridBackend(self.bm25, self.dense, 0.5)
        result = backend.search_group(["q1", "q2"])
        self.assertEqual(result.ranking, ["a", "b", "c"])
        premature_action_fusion = weighted_rrf(
            fuse_subqueries([self.bm25["q1"], self.bm25["q2"]]),
            fuse_subqueries([self.dense["q1"], self.dense["q2"]]),
            0.5,
        )
        self.assertEqual(premature_action_fusion, ["b", "a", "c"])
        self.assertNotEqual(result.ranking, premature_action_fusion)
        self.assertEqual(result.logical_search_calls, 2)
        self.assertEqual(result.underlying_search_calls, 4)

    def test_default_call_charges_do_not_disclose_endpoint_vs_interior(self):
        for weight in (0, 0.125, 0.5, 0.875, 1):
            with self.subTest(weight=weight):
                result = FrozenHybridBackend(
                    self.bm25, self.dense, weight
                ).search_group(["q1", "q2"])
                self.assertEqual(result.logical_search_calls, 2)
                self.assertEqual(result.underlying_search_calls, 4)

    def test_active_execution_charges_only_nonzero_weight_backends(self):
        for weight, expected in ((0, 2), (0.5, 4), (1, 2)):
            with self.subTest(weight=weight):
                result = FrozenHybridBackend(
                    self.bm25, self.dense, weight, execution="active"
                ).search_group(["q1", "q2"])
                self.assertEqual(result.logical_search_calls, 2)
                self.assertEqual(result.underlying_search_calls, expected)

    def test_endpoint_action_rankings_match_original_composition(self):
        for weight, original in ((0, self.bm25), (1, self.dense)):
            with self.subTest(weight=weight):
                backend = FrozenHybridBackend(self.bm25, self.dense, weight)
                for query in self.bm25:
                    self.assertEqual(backend.search(query), original[query])
                self.assertEqual(
                    backend.search_group(["q1", "q2"]).ranking,
                    fuse_subqueries([original["q1"], original["q2"]]),
                )

    def test_empty_groups_and_repeated_raw_calls_keep_correct_accounting(self):
        backend = FrozenHybridBackend(self.bm25, self.dense, 0.5)
        empty = backend.search_group([])
        self.assertEqual(empty.ranking, [])
        self.assertEqual((empty.logical_search_calls, empty.underlying_search_calls), (0, 0))
        repeated = backend.search_group(["q1", "q1"])
        self.assertEqual(repeated.ranking, backend.search("q1"))
        # Reusing a cached query does not make its nominal serving call free.
        self.assertEqual((repeated.logical_search_calls, repeated.underlying_search_calls), (2, 4))

    def test_missing_queries_and_misaligned_endpoint_caches_fail_closed(self):
        for weight in (0, 0.5, 1):
            with self.subTest(weight=weight):
                backend = FrozenHybridBackend(self.bm25, self.dense, weight)
                with self.assertRaises(KeyError):
                    backend.search("uncached")
                with self.assertRaises(KeyError):
                    backend.search_group(["q1", "uncached"])
        with self.assertRaises(ValueError):
            FrozenHybridBackend({"q1": ["a"]}, {"q2": ["b"]}, 0)
        with self.assertRaises(ValueError):
            FrozenHybridBackend(self.bm25, self.dense, 0.5, execution="unknown")


class ActionGroupingTests(unittest.TestCase):
    def test_valid_generation_keeps_two_decomposed_raw_queries(self):
        query = "How do cats sleep?"
        row = {
            "valid": True,
            "semantic_query": "cat sleeping behavior",
            "hypothetical_document": "Cats sleep in short episodes.",
            "subqueries": ["cat sleep duration", "cat sleep stages"],
        }
        self.assertEqual(
            action_queries(query, row),
            [[query], ["cats sleep"], [row["semantic_query"]],
             [row["hypothetical_document"]], row["subqueries"]],
        )

    def test_invalid_generation_keeps_unavailable_actions_empty(self):
        self.assertEqual(
            action_queries("How do cats sleep?", {"valid": False}),
            [["How do cats sleep?"], ["cats sleep"], [], [], []],
        )


if __name__ == "__main__":
    unittest.main()
