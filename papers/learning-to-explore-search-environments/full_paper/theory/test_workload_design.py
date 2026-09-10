"""Exact mathematical witnesses and independent enumeration checks."""
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workload_design as design


class WorkloadDesignTests(unittest.TestCase):
    def test_context_risks_and_nuisance_channel(self):
        utility, prior, channels = design.bit_problem(2, 3)
        self.assertEqual(design.risks(utility, prior), (F(1, 2), F(1, 2)))
        self.assertEqual(design.risk_matrix(utility, prior, channels),
                         ((0, F(1, 2), F(1, 2)), (F(1, 2), 0, F(1, 2))))

    def test_robust_design_hedges_task_mix_and_ignores_nuisance(self):
        utility, prior, channels = design.bit_problem(2, 3)
        rows = design.risk_matrix(utility, prior, channels)
        robust = design.minimax_design(rows, ((1, 0), (0, 1)))
        self.assertEqual(robust.risk, F(1, 4))
        self.assertEqual(robust.mixture, (F(1, 2), F(1, 2), 0))
        self.assertEqual(robust.workload_witness, (F(1, 2), F(1, 2)))
        nominal = design.minimax_design(rows, ((F(9, 10), F(1, 10)),))
        self.assertEqual(nominal.mixture, (1, 0, 0))
        self.assertEqual(nominal.risk, F(1, 20))
        self.assertEqual(design.universally_best(rows), ())
        self.assertEqual(design.crossing_pairs(rows), ((0, 1, 0, 1),))

    def test_universal_design_need_not_identify_the_world(self):
        utility, prior, channels = design.bit_problem(1, 3)
        rows = design.risk_matrix(utility, prior, channels)
        self.assertEqual(rows, ((0, F(1, 2)),))
        self.assertEqual(design.universally_best(rows), (0,))

    def test_workload_order_is_not_posterior_workload_order(self):
        example = design.robust_conditioning_example()
        self.assertEqual(example['before'], F(1, 4))
        self.assertEqual(example['fixed_workload_after'], F(1, 4))
        self.assertEqual(example['outcome_reacting_workload_after'], F(1, 2))
        self.assertEqual(example['incorrect_fixed_workload_voi'], F(-1, 4))

    def test_prior_risk_decreases_in_every_context(self):
        rng = random.Random(192)
        for _ in range(20):
            utility = [[[F(rng.randrange(11), 10) for _ in range(3)] for _ in range(2)] for _ in range(4)]
            channel = []
            for _ in range(4):
                p = F(rng.randrange(11), 10)
                channel.append((p, 1 - p))
            before = design.risks(utility, (F(1, 4),) * 4)
            after = design.residuals(utility, (F(1, 4),) * 4, channel)
            self.assertTrue(all(a <= b for a, b in zip(after, before)))

    def test_timing_formula_matches_complete_subset_games(self):
        for contexts in range(1, 5):
            workloads = tuple(tuple(int(i == j) for i in range(contexts)) for j in range(contexts))
            for budget in range(contexts + 1):
                subsets = tuple(combinations(range(contexts), budget))
                rows = tuple(tuple(F(int(x not in subset), 2) for subset in subsets) for x in range(contexts))
                self.assertEqual(design.minimax_design(rows, workloads).risk,
                                 design.timing_risk(contexts, budget))

    def test_side_information_formula_matches_all_partitions(self):
        # Enumerate deterministic messages directly; empty message cells allowed.
        for contexts in range(1, 6):
            for messages in range(1, 4):
                for budget in (0, 1, 2):
                    best = F(1)
                    for assignment in product(range(messages), repeat=contexts):
                        sizes = [assignment.count(label) for label in range(messages)]
                        risk = max(max(F(0), F(1, 2) * (1 - F(budget, size))) for size in sizes if size)
                        best = min(best, risk)
                    self.assertEqual(best, design.timing_risk(contexts, budget, messages))

    def test_general_loss_matrix_retains_row_offsets(self):
        # Subtracting each row minimum would incorrectly report zero here.
        result = design.minimax_design(((F(1, 5), F(2, 5)), (F(1, 2), F(3, 5))), ((1, 0), (0, 1)))
        self.assertEqual(result.risk, F(1, 2))
        self.assertEqual(result.mixture, (1, 0))

    def test_small_random_games_have_valid_exact_certificates(self):
        rng = random.Random(781)
        for _ in range(30):
            contexts, options = rng.randrange(1, 4), rng.randrange(1, 4)
            rows = tuple(tuple(F(rng.randrange(11), 10) for _ in range(options)) for _ in range(contexts))
            vertices = tuple(tuple(int(x == k) for x in range(contexts)) for k in range(contexts))
            answer = design.minimax_design(rows, vertices)
            primal = max(sum(p * value for p, value in zip(answer.mixture, row)) for row in rows)
            dual = min(sum(answer.workload_witness[x] * rows[x][q] for x in range(contexts)) for q in range(options))
            self.assertEqual(primal, answer.risk)
            self.assertEqual(dual, answer.risk)

    def test_invalid_models_and_budget_contracts(self):
        for value in ([], [[[1, 0]], [[1, 0], [0, 1]]], [[[2, 0]]]):
            with self.assertRaises(ValueError):
                design.risks(value, (1,))
        with self.assertRaises(ValueError):
            design.minimax_design(((0, 1),), ())
        with self.assertRaises(ValueError):
            design.minimax_design(((0, 2),), ((1,),))
        with self.assertRaises(ValueError):
            design.minimax_design(((0, 1),), ((F(1, 2),),))
        for arguments in ((0, 1, 1), (2, -1, 1), (2, 1, 0), (2, True, 1)):
            with self.assertRaises(ValueError):
                design.timing_risk(*arguments)


if __name__ == '__main__':
    unittest.main()
