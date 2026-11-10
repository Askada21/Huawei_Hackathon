"""Small tests for the temporal and probability contracts."""
import unittest
import numpy as np
from common import split, cutoff, priors, prior_predict,calibration_training_mask

class Contracts(unittest.TestCase):
    def test_calibration_excludes_late_first_events(self):
        a={'time':np.array([cutoff(24),cutoff(25),cutoff(26),cutoff(26)+60]),
           'user':np.array([1,1,2,1])}
        _,early=split(a,25);_,later=split(a,26)
        self.assertGreater(len(np.intersect1d(early,later)),0)
        clean=early[calibration_training_mask(a,early,26)]
        np.testing.assert_array_equal(clean,[1])
        self.assertEqual(len(np.intersect1d(clean,later)),0)
    def test_first_event_and_no_future_labels(self):
        a={'time':np.array([cutoff(24),cutoff(24)+60,cutoff(25),cutoff(25)+60,cutoff(26)]),
           'user':np.array([1,2,1,1,2]),'item':np.tile(np.arange(1,11),(5,1)),
           'label':np.zeros((5,10),dtype=np.int8)}
        a['label'][0,0]=1
        fit,v=split(a,25)
        np.testing.assert_array_equal(fit,[0,1]);np.testing.assert_array_equal(v,[2,4])
        s=priors(a,fit,3,11);p=prior_predict(a,v,s)
        a['label'][2:,1]=1
        p2=prior_predict(a,v,priors(a,fit,3,11))
        np.testing.assert_array_equal(p,p2)
        self.assertTrue(np.isfinite(p).all() and (p>=0).all() and (p<=1).all())

if __name__=='__main__':unittest.main()
