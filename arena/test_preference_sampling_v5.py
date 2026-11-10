import unittest
import numpy as np
from preference_sampling_v5 import observed_negative_pool

class NegativePoolContracts(unittest.TestCase):
    def test_excludes_clicks_and_later_rows(self):
        a={'user':np.array([1,2,1]),'item':np.arange(30).reshape(3,10),
           'label':np.zeros((3,10),dtype=int)}
        a['label'][0,3]=1;a['label'][1,2]=1
        items,n,start=observed_negative_pool(a,np.array([0,1]),4)
        self.assertNotIn(3,items);self.assertNotIn(12,items)
        self.assertTrue((items<20).all())
        np.testing.assert_array_equal(n,[0,9,9,0])
        self.assertTrue((items[start[1]:start[1]+n[1]]<10).all())
        self.assertTrue((items[start[2]:start[2]+n[2]]>=10).all())
if __name__=='__main__':unittest.main()
