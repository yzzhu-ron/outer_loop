import unittest
import numpy as np
from experiment import branches, fit, gains, onboard, outcome, route, rrf, ndcg

class Experiments(unittest.TestCase):
    def xor_model(self):
        utility=np.zeros((4,4,2)); likelihood=np.zeros((4,2,2))
        for w in range(4):
            a,b=divmod(w,2);utility[w,:,a^b]=1
            likelihood[w,0,a]=1;likelihood[w,1,b]=1
        return {'names':list('abcd'),'utilities':utility,'frequencies':np.full(4,.25),'likelihood':likelihood}

    def test_two_complementary_channels_have_joint_value(self):
        m=self.xor_model();p=np.full(4,.25)
        one=gains(m,p,[0,1],'myopic_value',{0:1,1:1},2)
        two=gains(m,p,[0,1],'lookahead2',{0:1,1:1},2)
        self.assertTrue(all(abs(x)<1e-12 for x in one.values()))
        self.assertTrue(all(abs(x-.5)<1e-12 for x in two.values()))
        self.assertEqual(gains(m,p,[0,1],'lookahead2',{0:1,1:1},1),one)

    def test_callback_only_selected_and_group_removed(self):
        m={'names':['a','b'],'utilities':np.array([[[1.,0.]]*4,[[0.,1.]]*4]),'frequencies':np.full(4,.25),'likelihood':np.full((2,8,5),.2)}
        candidates=[{'id':str(i),'family':'direct_question','action':1+i%4,'cost':2,'question_group':str(i//2),'bucket':0} for i in range(8)]
        called=[]
        def observe(i):called.append(i);return {'delta':0}
        for method in ('random','coverage','information_gain','decision_entropy','myopic_value','lookahead2'):
            called.clear()
            p,trace,spent=onboard(m,candidates,observe,method,5,17,{'likelihood_power':1})
            self.assertEqual(spent,4);self.assertEqual(called,[r['id'] for r in trace])
            self.assertEqual(len({int(i)//2 for i in called}),len(called))
        dirty=candidates[0]|{'scores':[1,2]}
        with self.assertRaises(ValueError):onboard(m,[dirty],observe,'random',2,0,{'likelihood_power':1})

    def test_source_fit_does_not_read_test(self):
        class Tasks(dict):
            def __getitem__(self,k):
                if k!='train':raise AssertionError('test leakage')
                return super().__getitem__(k)
        train={'scores':[[1.,0.],[0.,1.]],'buckets':[0,1]}
        sources=[{'name':'x','tasks':Tasks(train=train),'pools':{}}]
        model=fit(sources,{'utility_shrinkage':8,'dirichlet_per_outcome':.5})
        self.assertTrue(np.allclose(model['likelihood'],.2))
        self.assertEqual(route(model,np.array([1.])).tolist(),[0,1,0,0])

    def test_outcome_boundaries_and_invalid(self):
        self.assertEqual([outcome(v) for v in (-1,-.25,0,.25,1)],[0,1,2,3,4])
        for value in (float('nan'),float('inf'),1.001):
            with self.assertRaises(ValueError):outcome(value)

    def test_fusion_and_graded_gain(self):
        self.assertEqual(rrf([['b','a'],['a','b']]),['a','b'])
        self.assertAlmostEqual(ndcg(['a','b'],{'a':2,'b':1}),1)
        self.assertLess(ndcg(['b','a'],{'a':2,'b':1}),1)

    def test_posterior_branches_conserve_prior(self):
        m=self.xor_model();p=np.array([.1,.2,.3,.4])
        children=branches(m,p,0)
        self.assertAlmostEqual(sum(a for a,b in children),1)
        self.assertTrue(np.allclose(sum(a*b for a,b in children),p))

if __name__=='__main__':unittest.main()
