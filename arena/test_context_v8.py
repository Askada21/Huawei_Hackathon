import unittest
import numpy as np
from history_model_v3 import HistoryFactors,torch
class ContextContracts(unittest.TestCase):
    def setUp(self):
        self.stats=(np.ones(3)*20,np.ones(3)*2,np.ones(8)*10,np.ones(8),.1)
    def test_neutral_transform_preserves_existing_logits(self):
        torch.manual_seed(42);base=HistoryFactors(3,8,4,self.stats,gates=True,gate_mode='global')
        new=HistoryFactors(3,8,4,self.stats,gates=True,gate_mode='global',context_rank=2)
        new.load_state_dict(base.state_dict(),strict=False)
        u=torch.tensor([1,2]);i=torch.tensor([[1,2],[3,4]]);h=torch.tensor([[2,3,0],[0,0,0]])
        b=torch.tensor([0,1]);d=torch.tensor([0,1])
        torch.testing.assert_close(base.score(u,i,h,b,d),new.score(u,i,h,b,d),rtol=0,atol=0)
    def test_only_selected_context_changes(self):
        m=HistoryFactors(3,8,4,self.stats,context_rank=2)
        with torch.no_grad():m.context_left.fill_(1);m.context_right[0].fill_(.1)
        p=torch.ones((2,4));out=m.contextualize(p,torch.tensor([0,1]))
        torch.testing.assert_close(out[1],p[1]);self.assertGreater(out[0].sum(),p[0].sum())
        out.sum().backward();self.assertTrue(torch.isfinite(m.context_right.grad).all())
    def test_temporal_gate_preserves_initial_context_basis(self):
        torch.manual_seed(42);base=HistoryFactors(3,8,4,self.stats,gates=True,context_rank=2)
        torch.manual_seed(42);new=HistoryFactors(3,8,4,self.stats,gates=True,context_rank=2,temporal_gates=True)
        torch.testing.assert_close(base.context_left,new.context_left,rtol=0,atol=0)
        torch.testing.assert_close(new.temporal_gate.weight,torch.zeros((4,2)),rtol=0,atol=0)
        u=torch.tensor([1,2]);i=torch.tensor([[1,2],[3,4]]);h=torch.tensor([[2,3,0],[0,0,0]])
        b=torch.tensor([0,1]);d=torch.tensor([0,1])
        before=base.score(u,i,h,b,d)
        torch.testing.assert_close(before,new.score(u,i,h,b,d),rtol=0,atol=0)
        with torch.no_grad():new.temporal_gate.weight[0,0]=1
        after=new.score(u,i,h,b,d)
        torch.testing.assert_close(before[1],after[1],rtol=0,atol=0)
        self.assertGreater((after[0]-before[0]).abs().max().item(),0)
        after.sum().backward()
        self.assertTrue(torch.isfinite(new.temporal_gate.weight.grad).all())
    def test_temporal_gate_requires_enabled_gates(self):
        with self.assertRaisesRegex(ValueError,'requires gates'):
            HistoryFactors(3,8,4,self.stats,temporal_gates=True)
if __name__=='__main__':unittest.main()
