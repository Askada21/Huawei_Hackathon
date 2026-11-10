import unittest
import numpy as np
from history_model_v3 import HistoryFactors,torch

class TimeHistoryContracts(unittest.TestCase):
    def setUp(self):
        self.stats=(np.ones(3)*20,np.ones(3)*2,np.ones(8)*10,np.ones(8),.1)
        self.inputs=(torch.tensor([1,2]),torch.tensor([[1,2],[3,4]]),
                     torch.tensor([[2,3,0],[0,0,0]]),torch.tensor([0,1]),torch.tensor([0,1]))
        self.ht=torch.tensor([[0,21600,0],[0,0,0]])
    def test_neutral_scores_and_rng(self):
        torch.manual_seed(42);base=HistoryFactors(3,8,4,self.stats,gates=True,context_rank=2)
        torch.manual_seed(42);new=HistoryFactors(3,8,4,self.stats,gates=True,context_rank=2,time_history=True)
        torch.testing.assert_close(base.context_left,new.context_left,rtol=0,atol=0)
        torch.testing.assert_close(base.score(*self.inputs),new.score(*self.inputs,history_time=self.ht),rtol=0,atol=0)
    def test_padding_and_empty_history_stay_neutral(self):
        m=HistoryFactors(3,8,4,self.stats,time_history=True)
        with torch.no_grad():m.history_bucket_logits[0,0]=1
        p=m.score(*self.inputs,history_time=self.ht)
        ht=self.ht.clone();ht[0,2]=21600;ht[1]=43200
        torch.testing.assert_close(p,m.score(*self.inputs,history_time=ht),rtol=0,atol=0)
        p.sum().backward();self.assertTrue(torch.isfinite(m.history_bucket_logits.grad).all())
        self.assertGreater(m.history_bucket_logits.grad.abs().max().item(),0)
    def test_observed_history_time_changes_only_relevant_event(self):
        m=HistoryFactors(3,8,4,self.stats,time_history=True)
        with torch.no_grad():m.history_bucket_logits[0,0]=1
        p=m.score(*self.inputs,history_time=self.ht)
        ht=self.ht.clone();ht[0,0]=21600
        q=m.score(*self.inputs,history_time=ht)
        self.assertGreater((p[0]-q[0]).abs().max().item(),0)
        torch.testing.assert_close(p[1],q[1],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
