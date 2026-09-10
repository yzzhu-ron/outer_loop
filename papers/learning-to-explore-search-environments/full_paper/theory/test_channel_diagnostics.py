"""Known channel orders and independently checkable separation witnesses."""
from unittest import TestCase, main
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from channel_diagnostics import decision_labels, garbling_distance, separation_certificate


class ChannelDiagnosticsTests(TestCase):
    def test_identity_and_erasure_are_directional(self):
        identity = np.eye(2)
        erasure = np.full((2, 2), .5)
        self.assertLess(garbling_distance(identity, erasure)['verified_primal_linf_residual'], 1e-12)
        reverse = garbling_distance(erasure, identity)
        self.assertAlmostEqual(reverse['lp_linf_distance'], .5)
        self.assertEqual(reverse['rational_separation']['lower_bound_fraction'], '1/2')
        self.assertLess(garbling_distance(identity, identity)['verified_primal_linf_residual'], 1e-12)

    def test_different_bits_are_incomparable(self):
        first = np.eye(2)[[0, 0, 1, 1]]
        second = np.eye(2)[[0, 1, 0, 1]]
        for a, b in ((first, second), (second, first)):
            result = garbling_distance(a, b)
            self.assertAlmostEqual(result['lp_linf_distance'], .5)
            self.assertEqual(result['rational_separation']['lower_bound_fraction'], '1/2')

    def test_any_separation_weights_are_valid(self):
        a = np.array([[.75, .25], [.25, .75]])
        g = np.array([[.8, .2], [.4, .6]])
        b = a @ g
        for w in (np.array([[1, -1], [2, 3]]), np.zeros((2, 2))):
            self.assertLessEqual(separation_certificate(a, b, w)['lower_bound'], 1e-15)

    def test_unique_labels_and_merged_labels(self):
        model = {'names': ['a', 'b'], 'utilities': [[[1, 0]], [[0, 1]]]}
        self.assertTrue(decision_labels(model)['label_entropy_equals_world_entropy_for_every_posterior'])
        model['utilities'][1] = [[1, 0]]
        self.assertFalse(decision_labels(model)['label_entropy_equals_world_entropy_for_every_posterior'])
        self.assertEqual(decision_labels(model)['uniform_prior_label_masses'], [1.0])

    def test_solver_failure_is_never_incomparability(self):
        with patch('channel_diagnostics.linprog', return_value=SimpleNamespace(success=False, status=4, message='numerical failure')):
            with self.assertRaisesRegex(RuntimeError, 'status=4'):
                garbling_distance(np.eye(2), np.eye(2))

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            garbling_distance([[.2, .3]], [[.5, .5]])


if __name__ == '__main__':
    main()
