import unittest
import numpy as np
from experiment import select, measure, SUBSETS, CALLS

class Frontier(unittest.TestCase):
    def test_source_selection_respects_all_cost_dimensions(self):
        means=np.arange(len(SUBSETS),dtype=float)
        for cap in range(1,7):
            for gen in (False,True):
                i=select([means],cap,gen)
                s=SUBSETS[i]
                self.assertLessEqual(sum(CALLS[a] for a in s),cap)
                self.assertTrue(gen or max(s)<2)
        self.assertEqual(select([means],6,True),len(SUBSETS)-1)

    def test_singleton_preserves_backend_ranking_ties(self):
        task={'ids':['x'],'rankings':[[['b','a']]*5],'qrels':{'x':{'b':1,'a':0}}}
        scores=measure(task)
        self.assertEqual(scores.shape,(1,31))
        self.assertTrue(np.allclose(scores,1))

    def test_equal_family_mean_not_query_count_weighting(self):
        a=np.zeros(31);b=np.zeros(31);a[0]=1;b[1]=.9
        self.assertEqual(select([a,b],1,False),0)
        a[0]=.8
        self.assertEqual(select([a,b],1,False),1)

if __name__=='__main__':unittest.main()
