"""Mathematical regression tests; run unittest discover -s theory."""

import random
import unittest
from fractions import Fraction as F

from finite_decisions import (
    bayes_regret,
    deterministic_likelihood,
    environment_information,
    expected_regret_after,
    gaussian_variance_reduction,
    minimax_radius,
    pairwise_incompatibility,
    probe_value,
    robust_after_partition,
    select_action_from_contrasts,
    solve_linear,
)
from verify_theory import run


class FiniteDecisionTests(unittest.TestCase):
    def test_pairwise_zero_does_not_imply_global_compatibility(self):
        utilities = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
        result = minimax_radius(utilities)
        self.assertEqual(pairwise_incompatibility(utilities), 0)
        self.assertEqual(result.value, F(1, 3))
        self.assertEqual(result.action_mixture, (F(1, 3),) * 3)
        self.assertEqual(result.witness_prior, (F(1, 3),) * 3)

    def test_two_action_formula_including_zero_and_asymmetric_gaps(self):
        for alpha in (F(0), F(1, 5), F(1)):
            for beta in (F(0), F(1, 3), F(1)):
                radius = minimax_radius(((0, alpha), (beta, 0))).value
                expected = alpha * beta / (alpha + beta) if alpha + beta else F(0)
                self.assertEqual(radius, expected)

    def test_multiway_conflict_persists_without_ties(self):
        for epsilon in (F(1, 100), F(1, 10), F(1, 2), F(1)):
            utilities = ((0, 1, 1 - epsilon), (1 - epsilon, 0, 1), (1, 1 - epsilon, 0))
            self.assertTrue(all(row.count(max(row)) == 1 for row in utilities))
            self.assertEqual(pairwise_incompatibility(utilities), epsilon)
            self.assertEqual(minimax_radius(utilities).value, (1 + epsilon) / 3)

    def test_exact_primal_dual_and_pair_bound_on_random_games(self):
        rng = random.Random(834)
        for n, m in ((1, 3), (3, 1), (2, 3), (3, 2), (4, 3)):
            for _ in range(12):
                utilities = [[F(rng.randrange(5), 4) for _ in range(m)] for _ in range(n)]
                result = minimax_radius(utilities)
                self.assertEqual(sum(result.action_mixture), 1)
                self.assertEqual(sum(result.witness_prior), 1)
                self.assertEqual(bayes_regret(utilities, result.witness_prior), result.value)
                self.assertLessEqual(sum(w > 0 for w in result.witness_prior), m)
                self.assertLessEqual(pairwise_incompatibility(utilities) / 2, result.value)
                self.assertGreaterEqual(result.value, 0)
                self.assertLessEqual(result.value, 1)

    def test_partitions_cannot_increase_robust_radius(self):
        utilities = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
        coarse = robust_after_partition(utilities, [0, 0, 0])
        refined = robust_after_partition(utilities, [0, 0, 1])
        self.assertEqual(coarse, F(1, 3))
        self.assertEqual(refined, 0)

    def test_perfect_synthetic_score_can_leave_maximal_binary_regret(self):
        utilities = ((1, 0), (0, 1))
        self.assertEqual(robust_after_partition(utilities, [1, 1]), F(1, 2))
        self.assertEqual(robust_after_partition(utilities, [0, 1]), 0)

    def test_independent_noise_has_zero_value(self):
        utilities, prior = ((1, 0), (0, 1)), (F(1, 2), F(1, 2))
        noise = ((F(1, 2), F(1, 2)),) * 2
        self.assertEqual(probe_value(utilities, prior, noise), 0)
        self.assertAlmostEqual(environment_information(prior, noise), 0)

    def test_probe_value_matches_change_in_best_expected_utility(self):
        utilities, prior = ((1, 0), (0, 1)), (F(1, 2), F(1, 2))
        noisy_signal = ((F(1, 5), F(4, 5)), (0, 1))
        self.assertEqual(expected_regret_after(utilities, prior, noisy_signal), F(2, 5))
        self.assertEqual(probe_value(utilities, prior, noisy_signal), F(1, 10))

    def test_complementary_probes_defeat_myopic_stopping(self):
        result = run()["complementarity"]
        self.assertEqual(result["one_step_values"], [0, 0])
        self.assertEqual(result["two_probe_value"], F(1, 2))

    def test_hardness_and_entropy_can_both_choose_irrelevant_probes(self):
        result = run()["one_probe_acquisition"]
        self.assertEqual(result["selected"]["hardest"], "hard_independent")
        self.assertEqual(result["selected"]["environment_entropy"], "nuisance_signature")
        self.assertEqual(result["selected"]["decision_value"], "easy_aligned")
        self.assertEqual(result["selected_regret"]["decision_value"], F(2, 5))
        self.assertEqual(result["uniform_random_expected_regret"], F(7, 15))

    def test_gaussian_surrogate_ignores_orthogonal_nuisance(self):
        self.assertEqual(gaussian_variance_reduction(((100, 0), (0, 1)), (1, 0), (0, 1), 1), 0)
        self.assertEqual(gaussian_variance_reduction(((100, 0), (0, 1)), (0, 1), (0, 1), 1), F(1, 2))

    def test_zero_probability_observations_and_degenerate_priors(self):
        utilities, prior = ((1, 0), (0, 1)), (1, 0)
        self.assertEqual(probe_value(utilities, prior, ((1, 0, 0), (0, 1, 0))), 0)
        self.assertEqual(minimax_radius(((1, 1),)).value, 0)

    def test_invalid_probabilities_fail_explicitly(self):
        with self.assertRaises(ValueError):
            probe_value(((1, 0), (0, 1)), (F(1, 2), F(1, 2)), ((1, 1), (0, 1)))
        with self.assertRaises(ValueError):
            bayes_regret(((1, 0),), (F(1, 2),))

    def test_linear_system_rejects_missing_extra_and_ragged_rows(self):
        for coefficients, rhs in (((), ()), (((1,), (100,)), (1,)), (((1,),), (1, 2)),
                                  (((1, 2), (3,)), (1, 2))):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                solve_linear(coefficients, rhs)

    def test_negative_noise_variance_cannot_create_excess_variance_reduction(self):
        with self.assertRaisesRegex(ValueError, "Noise variance"):
            gaussian_variance_reduction(((1,),), (1,), (1,), F(-1, 2))

    def test_invalid_covariances_are_rejected(self):
        for covariance in (((1, 2), (0, 1)), ((1, 2), (2, 1)), ((0, 1), (1, 1)), ((-1, 0), (0, 1))):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                gaussian_variance_reduction(covariance, (1, 0), (1, 0), 2)

    def test_singular_psd_covariance_is_allowed(self):
        self.assertEqual(gaussian_variance_reduction(((1, 1), (1, 1)), (1, 0), (0, 1), 1), F(1, 2))
        self.assertEqual(gaussian_variance_reduction(((0, 0), (0, 1)), (1, 0), (0, 1), 1), 0)

    def test_reference_action_remains_available_with_negative_contrasts(self):
        self.assertEqual(select_action_from_contrasts((-1, F(-1, 2))), 0)
        self.assertEqual(select_action_from_contrasts((0, -1)), 0)
        self.assertEqual(select_action_from_contrasts((F(1, 2), -1)), 1)
        result = run()["reference_router_alignment"]
        self.assertEqual(result["selected_action_including_reference"], 0)
        self.assertEqual(result["regret_if_reference_is_omitted"], 1)

    def test_kernel_counterexample_stays_in_bounded_domain_interior(self):
        result = run()["bounded_domain_nonidentifiability"]
        self.assertTrue(all(0 < x < 1 for point in result["interior_parameter_points"] for x in point))
        self.assertEqual(result["probe_means"], (F(1, 2), F(1, 2)))
        self.assertEqual(result["nonreference_contrasts"], (F(-1, 4), F(1, 4)))
        self.assertEqual(result["optimal_actions"], (0, 1))


if __name__ == "__main__":
    unittest.main()
