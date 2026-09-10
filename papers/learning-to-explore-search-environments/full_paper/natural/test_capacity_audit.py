import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from capacity_audit import audit

class Capacity(unittest.TestCase):
    def test_joint_posterior_cannot_choose_contexts_independently(self):
        utility=np.array([[[1,0],[1,0]],[[0,1],[0,1]]],dtype=float)
        result=audit(utility,[0,1],[[1,0],[0,1]])
        self.assertEqual(result['relaxed_upper_bound']['ndcg'],1)
        self.assertEqual(result['verified_witness_lower_bound']['ndcg'],.5)
        self.assertEqual(result['bracket_width'],.5)

    def test_numerical_failure_is_not_infeasibility(self):
        with patch('capacity_audit.linprog',return_value=SimpleNamespace(success=False,status=4)):
            with self.assertRaises(RuntimeError):
                audit([[[1,0]],[[0,1]]],[0],[[1,0]])

    def test_reachable_policy_matches_oracle(self):
        utility=np.array([[[1,0],[0,1]],[[0,1],[1,0]]],dtype=float)
        result=audit(utility,[0,1],[[1,0],[0,1]])
        self.assertEqual(result['relaxed_upper_bound']['ndcg'],1)
        self.assertEqual(result['verified_witness_lower_bound']['ndcg'],1)
        self.assertEqual(result['bracket_width'],0)

if __name__=='__main__':unittest.main()
