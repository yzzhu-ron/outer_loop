"""Mechanical and independently derived analytical checks before protocol freeze."""
from collections import Counter
from fractions import Fraction as F
import unittest

from experiment import FiniteModel, evaluate, incomparability_witness


class ExactModelTests(unittest.TestCase):
    def tearDown(self):
        FiniteModel.clear_caches()

    def test_normalization_independence_and_canonical_history(self):
        model = FiniteModel("xor")
        for rows in model.likelihoods:
            self.assertTrue(all(sum(row) == 1 for row in rows))
        first = model.extend(model.extend((), 0, 1), 1, 0)
        second = model.extend(model.extend((), 1, 0), 0, 1)
        self.assertEqual(first, second)
        self.assertEqual(sum(model.posterior(first)), 1)
        self.assertEqual(model.marginals(((0, 1),)), (F(1), F(1, 2), F(1, 2)))
        with self.assertRaises(ValueError):
            model.extend(first, 0, 1)

    def test_initial_voi_and_incomparable_channels(self):
        model = FiniteModel("xor", F(3, 4))
        expected = {"d0": F(1, 16), "d1": F(1, 16), "r": F(0), "s": F(0), "nuisance": F(0), "proxy": F(3, 16)}
        for q, name in enumerate(model.probes):
            self.assertEqual(model.one_step_value((), q), expected[name])
            for other in range(len(model.probes)):
                if other != q:
                    self.assertIsNotNone(incomparability_witness(model, q, other))
        self.assertAlmostEqual(model.information((), model.probes.index("nuisance")), 2.0)

    def test_xor_complementarity_and_greedy_proxy_trap(self):
        model = FiniteModel("xor", F(1))
        self.assertEqual(model.dynamic_program((), 2)[0], 1)
        self.assertEqual(model.subset_value((), model.fixed_subset(2)), 1)
        greedy = evaluate(model, model, "decision_myopic", 2)
        planned = evaluate(model, model, "decision_lookahead_2", 2)
        self.assertEqual(F(greedy["expected_utility_exact"]), F(3, 4))
        self.assertEqual(F(greedy["expected_operations_exact"]), 1)
        self.assertEqual(F(planned["expected_utility_exact"]), 1)
        self.assertEqual(F(planned["expected_operations_exact"]), 2)

    def test_gated_adaptive_plan_beats_best_fixed_pair(self):
        model = FiniteModel("gated", F(3, 4))
        self.assertEqual(model.dynamic_program((), 2)[0], F(7, 8))
        self.assertEqual(model.subset_value((), model.fixed_subset(2)), F(3, 4))
        self.assertEqual(model.probes[model.dynamic_program((), 2)[2]], "gate")

    def test_shuffled_labels_preserve_marginals_and_observation_law(self):
        original = FiniteModel("xor")
        shuffled = FiniteModel("xor", shuffle_seed=101)
        self.assertEqual(Counter(original.labels), Counter(shuffled.labels))
        self.assertNotEqual(original.labels, shuffled.labels)
        self.assertEqual(original.likelihoods, shuffled.likelihoods)
        self.assertEqual(shuffled.value(), F(1, 2))

    def test_candidate_priority_never_reindexes_observation_channels(self):
        canonical = FiniteModel("xor")
        reverse = FiniteModel("xor", probe_priority="reverse")
        self.assertEqual(canonical.probes, reverse.probes)
        self.assertEqual(canonical.likelihoods, reverse.likelihoods)
        self.assertEqual(canonical.available(()), reverse.available(())[:: -1])
        self.assertEqual(canonical.posterior(((0, 1),)), reverse.posterior(((0, 1),)))

    def test_maximum_noise_makes_every_observation_uninformative(self):
        source = FiniteModel("gated")
        truth = FiniteModel("gated", extra_noise=F(1, 2))
        for rows in truth.likelihoods:
            self.assertTrue(all(row == rows[0] for row in rows))
        for method in ("random", "decision_myopic", "decision_lookahead_2"):
            result = evaluate(source, truth, method, 2)
            self.assertEqual(F(result["expected_utility_exact"]), F(1, 2))


if __name__ == "__main__":
    unittest.main()
