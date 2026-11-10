import unittest
from objectives_v5 import torch,event_softmax,event_pairwise,observed_global_rank

class PreferenceContracts(unittest.TestCase):
    def test_null_gradients_match_for_every_outcome(self):
        for target in range(11):
            y=torch.zeros((1,10),dtype=torch.double)
            if target<10:y[0,target]=1
            a=torch.zeros((1,10),dtype=torch.double,requires_grad=True)
            b=a.detach().clone().requires_grad_()
            event_softmax(a,y).backward();event_pairwise(b,y).backward()
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)

    def test_correct_winner_has_smaller_loss(self):
        y=torch.zeros((2,10));y[0,3]=1
        good=torch.full((2,10),-3.);good[0,3]=3
        bad=-good
        self.assertLess(event_pairwise(good,y).item(),event_pairwise(bad,y).item())

    def test_no_click_events_have_nonzero_gradient(self):
        z=torch.ones((4,10),requires_grad=True);y=torch.zeros_like(z)
        event_pairwise(z,y).backward()
        self.assertTrue((z.grad>0).all())

    def test_candidate_permutation_invariant(self):
        torch.manual_seed(42);z=torch.randn(3,10);y=torch.zeros_like(z);y[0,4]=1;y[2,2]=1
        p=torch.randperm(10)
        torch.testing.assert_close(event_pairwise(z,y),event_pairwise(z[:,p],y[:,p]))

    def test_global_ranking_uses_no_click_negatives(self):
        y=torch.zeros((3,10));y[0,1]=1
        z=torch.zeros_like(y,requires_grad=True)
        loss=observed_global_rank(z,y);loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad[1:].abs().sum().item(),0)

    def test_global_ranking_empty_positive_batch_is_finite(self):
        z=torch.ones((2,10),requires_grad=True)
        loss=observed_global_rank(z,torch.zeros_like(z));loss.backward()
        self.assertEqual(loss.item(),0)

if __name__=='__main__':unittest.main()
