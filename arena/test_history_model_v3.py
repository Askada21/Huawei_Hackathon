"""Contracts for masking, baseline equivalence, and history timing."""
import unittest
import numpy as np
from history_model_v3 import HistoryFactors,Factors,torch,selection_mask,cutoff,select_basis

class HistoryContracts(unittest.TestCase):
    def test_distributed_basis_is_paired_deterministic_and_excludes_leading(self):
        su=np.array([[1,1,1,0],[1,0,1,1],[1,0,1,1],[1,0,1,1]],dtype=float)
        si=np.array([[1,1,1,0],[1,0,1,1],[1,0,1,1],[1,0,1,1],[1,0,1,1]],dtype=float)
        # Component0 has greatest participation but must never be selected.
        u,i,meta=select_basis(su,si,2,'distributed')
        self.assertEqual(meta['selected_indices'],[2,3])
        np.testing.assert_array_equal(u,su[:,[2,3]])
        np.testing.assert_array_equal(i,si[:,[2,3]])
        _,_,again=select_basis(su*3,si*5,2,'distributed')
        self.assertEqual(meta['selected_indices'],again['selected_indices'])
        lu,li,lmeta=select_basis(su,si,2,'leading')
        self.assertEqual(lmeta['selected_indices'],[1,2])
        np.testing.assert_array_equal(lu,su[:,1:3])
        np.testing.assert_array_equal(li,si[:,1:3])

    def test_selection_excludes_late_first_events(self):
        a={'time':np.array([cutoff(25),cutoff(26)-1,cutoff(26),cutoff(26)+60])}
        events=np.array([0,1,2,3])
        np.testing.assert_array_equal(selection_mask(a,events,26),[True,True,False,False])
        np.testing.assert_array_equal(selection_mask(a,events),[True,True,True,True])

    def setUp(self):
        torch.manual_seed(42)
        self.stats=(np.array([0,20,40]),np.array([0,2,4]),np.ones(8)*10,np.ones(8),.1)
        self.u=torch.tensor([1,2]);self.i=torch.tensor([[1,2,3],[4,5,6]])
        self.h=torch.tensor([[1,3,0],[0,0,0]])
        self.b=torch.tensor([0,1]);self.day=torch.tensor([0,1])
        self.t=torch.tensor([1000000,1000000]);self.ht=torch.tensor([[900000,990000,0],[0,0,0]])

    def test_baseline_score_equivalence(self):
        base=Factors(3,8,4,self.stats);model=HistoryFactors(3,8,4,self.stats)
        model.load_state_dict(base.state_dict(),strict=False)
        with torch.no_grad():
            base.i.weight[0].zero_();model.i.weight[0].zero_()
            torch.testing.assert_close(base.score(self.u,self.i,self.h,self.b,self.day),
                model.score(self.u,self.i,self.h,self.b,self.day,self.t,self.ht))

    def test_masked_padding_and_empty_history(self):
        for mode in ['recency','candidate']:
            m=HistoryFactors(3,8,4,self.stats,mode=mode,separate_history=True,gates=True)
            with torch.no_grad():
                z=m.score(self.u,self.i,self.h,self.b,self.day,self.t,self.ht)
                m.context.weight[0].fill_(100)
                padded=self.ht.clone();padded[self.h==0]=999999
                zz=m.score(self.u,self.i,self.h,self.b,self.day,self.t,padded)
                torch.testing.assert_close(z,zz)
                self.assertTrue(torch.isfinite(z).all())

    def test_history_timestamps_required(self):
        m=HistoryFactors(3,8,4,self.stats,mode='recency')
        with self.assertRaises(ValueError):m.score(self.u,self.i,self.h,self.b,self.day)

    def test_global_gate_invariant_to_exposure_but_reliability_gate_is_not(self):
        for gate_mode in ['global','reliability']:
            m=HistoryFactors(3,8,4,self.stats,gates=True,gate_mode=gate_mode)
            with torch.no_grad():
                m.gate.weight.fill_(.7)
                before=m.score(self.u,self.i,self.h,self.b,self.day,self.t,self.ht)
                m.user_exposure[self.u]=100000
                after=m.score(self.u,self.i,self.h,self.b,self.day,self.t,self.ht)
            if gate_mode=='global':torch.testing.assert_close(before,after)
            else:self.assertFalse(torch.allclose(before,after))

if __name__=='__main__':unittest.main()
