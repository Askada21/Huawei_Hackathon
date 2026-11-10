import unittest
import numpy as np
from history_model_v3 import HistoryFactors,torch

class ProfileShrinkageContracts(unittest.TestCase):
    def setUp(self):
        self.stats=(np.array([0,10,100]),np.array([0,1,10]),np.ones(8)*10,np.ones(8),.1)
        self.u=torch.tensor([0,1,2]);self.i=torch.tensor([[1,2]]*3)
        self.h=torch.tensor([[3,4,0]]*3);self.b=torch.tensor([0,0,0]);self.day=self.b
    def test_fixed_confidence_is_convex_score_interpolation(self):
        base=HistoryFactors(3,8,4,self.stats,cold_history=True,gates=True,gate_mode='global',context_rank=2)
        m=HistoryFactors(3,8,4,self.stats,cold_history=True,gates=True,gate_mode='global',context_rank=2,profile_shrinkage=20)
        with torch.no_grad():base.hw.fill_(.2);base.cold_hw.fill_(.9);base.ut.weight.fill_(.3)
        m.load_state_dict(base.state_dict())
        known=base.score(self.u,self.i,self.h,self.b,self.day)
        cold=base.score(self.u,self.i,self.h,self.b,self.day,user_keep=torch.zeros(3,1,dtype=torch.bool))
        w=torch.tensor([0,10/30,100/120])[:,None]
        actual=m.score(self.u,self.i,self.h,self.b,self.day)
        torch.testing.assert_close(actual,w*known+(1-w)*cold,rtol=1e-5,atol=1e-6)
    def test_masking_removes_identity_and_gradients_remain_finite(self):
        m=HistoryFactors(3,8,4,self.stats,cold_history=True,profile_shrinkage=20,gates=True)
        p=m.score(self.u,self.i,self.h,self.b,self.day,user_keep=torch.zeros(3,1,dtype=torch.bool))
        torch.testing.assert_close(p[0],p[1],rtol=0,atol=0)
        torch.testing.assert_close(p[0],p[2],rtol=0,atol=0)
        p.sum().backward()
        torch.testing.assert_close(m.u.weight.grad,torch.zeros_like(m.u.weight.grad))
        self.assertTrue(torch.isfinite(m.cold_hw.grad))
    def test_requires_cold_path(self):
        with self.assertRaises(ValueError):HistoryFactors(3,8,4,self.stats,profile_shrinkage=20)
if __name__=='__main__':unittest.main()
