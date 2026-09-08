import unittest

from shopgym.search import run_grol


class ShopGymSmokeTests(unittest.TestCase):
    def test_mock_search_uses_shared_outer_loop_primitives(self):
        run = run_grol(
            task_category="product_search",
            n_eval=8,
            n_trace=3,
            max_steps=2,
            rng_seed=7,
        )
        self.assertEqual(2, len(run.steps))
        self.assertGreater(run.total_episodes, 8)


if __name__ == "__main__":
    unittest.main()
