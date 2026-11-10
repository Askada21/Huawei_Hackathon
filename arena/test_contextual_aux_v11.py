import unittest
import numpy as np
from preference_sampling_v5 import observed_bucket_negative_pool
from history_model_v3 import HistoryFactors,torch

class ContextualAuxContracts(unittest.TestCase):
    def test_real_negatives_same_user_bucket_past_only(self):
        a={'user':np.array([1,1,2,1]),'bucket':np.array([0,1,0,0]),
           'item':np.arange(40).reshape(4,10),'label':np.zeros((4,10),dtype=int)}
        a['label'][0,3]=1
        items,n,start=observed_bucket_negative_pool(a,np.array([0,1,2]),4)
        np.testing.assert_array_equal(items[start[4]:start[4]+n[4]],[0,1,2,4,5,6,7,8,9])
        np.testing.assert_array_equal(items[start[5]:start[5]+n[5]],np.arange(10,20))
        np.testing.assert_array_equal(items[start[8]:start[8]+n[8]],np.arange(20,30))
        self.assertEqual(n[6],0);self.assertTrue((items<30).all())
    def test_auxiliary_gradient_reaches_only_selected_context(self):
        stats=(np.ones(3)*20,np.ones(3)*2,np.ones(8)*10,np.ones(8),.1)
        m=HistoryFactors(3,8,4,stats,context_rank=2,context_mode='bucket')
        u=torch.tensor([1]);b=torch.tensor([2]);av=m.contextualize(m.u(u),b)
        diff=(av*(m.i(torch.tensor([3]))-m.i(torch.tensor([4])))).sum()
        torch.nn.functional.softplus(-diff).backward()
        g=m.context_right.grad
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(g[2].abs().sum().item(),0)
        torch.testing.assert_close(g[[0,1,3]],torch.zeros_like(g[[0,1,3]]))
if __name__=='__main__':unittest.main()
