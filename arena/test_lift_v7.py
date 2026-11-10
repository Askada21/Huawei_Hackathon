import unittest
from unittest.mock import patch
import numpy as np
from lift_v7 import build,score
class LiftContracts(unittest.TestCase):
    def test_repeated_clicks_do_not_multiply_user_evidence(self):
        c=np.array([[1,1,1],[1,2,2],[2,1,1],[2,2,2],[3,1,3]])
        with patch('lift_v7.interactions',return_value=c):g=build({},np.array([0]),4,5,topk=4)
        with patch('lift_v7.interactions',return_value=np.r_[c,c]):gg=build({},np.array([0]),4,5,topk=4)
        np.testing.assert_allclose(g.toarray(),gg.toarray());np.testing.assert_array_equal(g.diagonal(),0)
        self.assertTrue(np.isfinite(g.data).all())
    def test_empty_history_neutral(self):
        from scipy import sparse
        a={'history':np.zeros((1,20),dtype=int),'item':np.ones((1,10),dtype=int)}
        np.testing.assert_array_equal(score(sparse.csr_matrix((5,5)),a,np.array([0])),0)
if __name__=='__main__':unittest.main()
