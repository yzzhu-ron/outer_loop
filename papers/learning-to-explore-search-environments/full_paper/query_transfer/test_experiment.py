import unittest
import numpy as np
from experiment import QueryModel, candidates, fit_and_select, normalized_embeddings, resource_means


def features(axis, count):
    x = np.zeros((count, 392))
    x[:, 8 + axis] = 1
    return x


def part(axis, scores, count=4):
    return {"features": features(axis, count), "scores": np.tile(scores, (count, 1))}


class QueryTransferTests(unittest.TestCase):
    def test_exact_grid_and_embedding_only_distance(self):
        grid = list(candidates())
        self.assertEqual(len(grid), 15)
        self.assertEqual(len({row["name"] for row in grid}), 15)
        x = features(0, 1)
        modified = x.copy()
        modified[:, :8] = 10000
        np.testing.assert_array_equal(normalized_embeddings(x), normalized_embeddings(modified))

    def test_pooled_vs_balanced_and_global_shrinkage(self):
        training = [part(0, [1, 0, 0, 0, 0]), part(1, [0, 1, 0, 0, 0])]
        spec = {"kind": "knn", "k": 2, "shrinkage": 0.0, "balanced": False}
        pooled = QueryModel(spec).fit(training).predict(features(0, 1))
        balanced = QueryModel({**spec, "balanced": True}).fit(training).predict(features(0, 1))
        shrunk = QueryModel({**spec, "shrinkage": .5}).fit(training).predict(features(0, 1))
        np.testing.assert_allclose(pooled, [[1, 0, 0, 0, 0]])
        np.testing.assert_allclose(balanced, [[.5, .5, 0, 0, 0]])
        np.testing.assert_allclose(shrunk, [[.75, .25, 0, 0, 0]])

    def test_global_prior_balances_families_not_row_counts(self):
        training = [part(0, [1, 0, 0, 0, 0], 2), part(1, [0, 1, 0, 0, 0], 8)]
        model = QueryModel({"kind": "fixed"}).fit(training)
        np.testing.assert_allclose(model.prior, [.5, .5, 0, 0, 0])
        self.assertFalse(model.needs_embedding)

    def test_calibration_cannot_change_fitted_parameters(self):
        training = [part(0, [1, 0, 0, 0, 0]), part(1, [0, 1, 0, 0, 0])]
        calibration = [part(0, [0, 1, 0, 0, 0]), part(1, [1, 0, 0, 0, 0])]
        changed = [part(1, [0, 0, 1, 0, 0]), part(0, [0, 0, 0, 1, 0])]
        first, _, _ = fit_and_select(training, calibration)
        second, _, _ = fit_and_select(training, changed)
        for name in first:
            np.testing.assert_array_equal(first[name].predict(features(0, 1)), second[name].predict(features(0, 1)))
        np.testing.assert_allclose(first["ridge_10"].estimator[0].mean_, np.concatenate([row["features"] for row in training]).mean(axis=0))

    def test_fusion_counts_shared_generation_once(self):
        row = [{"search_calls": 1, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0} for _ in range(2)]
        row += [{"search_calls": calls, "llm_calls": 1, "input_tokens": 400, "output_tokens": 120} for calls in (1, 1, 2)]
        fused = resource_means([row])
        self.assertEqual(fused, {"search_calls": 6.0, "llm_calls": 1.0, "input_tokens": 400.0, "output_tokens": 120.0})
        self.assertEqual(resource_means([row], [0])["llm_calls"], 0)
        row[3]["input_tokens"] = 500
        with self.assertRaises(ValueError):
            resource_means([row])


if __name__ == "__main__":
    unittest.main()
