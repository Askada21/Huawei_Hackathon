import unittest
import numpy as np
from adapt_v7 import adapt,probabilities
class AdaptContracts(unittest.TestCase):
    def test_repeated_negative_impressions_reduce_score(self):
        x=np.ones((3,10,1));z=np.zeros((3,10));y=np.zeros_like(z)
        d=adapt(x,z,y,np.array([20.]))
        self.assertLess(d[0],0);self.assertTrue((probabilities(z+x@d)<probabilities(z)).all())
    def test_large_ridge_retains_original_scores(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(4,10,3));z=rng.normal(size=(4,10));y=np.zeros_like(z);y[0,2]=1
        d=adapt(x,z,y,np.full(3,1e12))
        np.testing.assert_allclose(probabilities(z+x@d),probabilities(z),atol=1e-9)
if __name__=='__main__':unittest.main()
