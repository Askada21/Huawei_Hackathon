import unittest
import numpy as np
from history_model_v3 import HistoryFactors,torch

class ColdHistoryContracts(unittest.TestCase):
    def setUp(self):
        self.stats=(np.array([0,20,40]),np.array([0,2,4]),np.ones(8)*10,np.ones(8),.1)
        self.u=torch.tensor([1,2]);self.i=torch.tensor([[1,2],[1,2]])
        self.h=torch.tensor([[3,4,0],[3,4,0]]);self.b=torch.tensor([0,0]);self.d=torch.tensor([0,0])
    def test_neutral_scores_and_initialization(self):
        torch.manual_seed(42);base=HistoryFactors(3,8,4,self.stats,gates=True,context_rank=2)
        torch.manual_seed(42);new=HistoryFactors(3,8,4,self.stats,gates=True,context_rank=2,cold_history=True)
        torch.testing.assert_close(base.context_left,new.context_left,rtol=0,atol=0)
        z=base.score(self.u,self.i,self.h,self.b,self.d)
        torch.testing.assert_close(z,new.score(self.u,self.i,self.h,self.b,self.d),rtol=0,atol=0)
        torch.testing.assert_close(z,new.score(self.u,self.i,self.h,self.b,self.d,user_keep=torch.ones(2,1,dtype=torch.bool)),rtol=0,atol=0)
    def test_hidden_user_identifiers_cannot_affect_scores(self):
        m=HistoryFactors(3,8,4,self.stats,gates=True,cold_history=True)
        with torch.no_grad():m.ut.weight[1].fill_(9);m.ub.weight[2].fill_(5)
        z=m.score(self.u,self.i,self.h,self.b,self.d,user_keep=torch.zeros(2,1,dtype=torch.bool))
        torch.testing.assert_close(z[0],z[1],rtol=0,atol=0)
        z.sum().backward()
        for parameter in [m.u.weight,m.ub.weight,m.ut.weight]:
            torch.testing.assert_close(parameter.grad,torch.zeros_like(parameter.grad))
        self.assertTrue(torch.isfinite(m.cold_hw.grad));self.assertGreater(m.cold_hw.grad.abs().item(),0)
    def test_cold_scale_affects_unseen_users_only(self):
        m=HistoryFactors(3,8,4,self.stats,cold_history=True)
        u=torch.tensor([0,1]);z=m.score(u,self.i,self.h,self.b,self.d)
        with torch.no_grad():m.cold_hw.add_(1)
        zz=m.score(u,self.i,self.h,self.b,self.d)
        torch.testing.assert_close(z[1],zz[1],rtol=0,atol=0)
        self.assertGreater((z[0]-zz[0]).abs().max().item(),0)
if __name__=='__main__':unittest.main()
