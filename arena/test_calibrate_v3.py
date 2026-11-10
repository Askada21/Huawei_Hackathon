"""One-day temporal and event-disjointness contracts for multi-period stacking."""
import unittest
import numpy as np
from calibrate_v3 import period_mask,check_unique_period
from common import cutoff

class MultiPeriodContracts(unittest.TestCase):
    def test_first_event_crossing_next_day_is_excluded(self):
        times=np.array([cutoff(24),cutoff(24)+60,cutoff(25),cutoff(26)])
        mask=period_mask(times,np.array([0,1,2,3]),24,26)
        np.testing.assert_array_equal(mask,[True,True,False,False])

    def test_outer_period_rejected(self):
        with self.assertRaisesRegex(ValueError,'outer'):
            period_mask(np.array([cutoff(26)]),np.array([0]),26,26)

    def test_duplicate_period_rejected(self):
        days=set();events=set();check_unique_period(24,np.array([1,2]),days,events,np.array([8]))
        with self.assertRaisesRegex(ValueError,'date'):
            check_unique_period(24,np.array([3]),days,events,np.array([8]))

    def test_duplicate_event_across_periods_rejected(self):
        days=set();events=set();check_unique_period(24,np.array([1,2]),days,events,np.array([8]))
        with self.assertRaisesRegex(ValueError,'Duplicate training'):
            check_unique_period(25,np.array([2,3]),days,events,np.array([8]))

    def test_overlap_with_outer_events_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            check_unique_period(25,np.array([7,8]),set(),set(),np.array([8,9]))

    def test_extended_windows_rejected(self):
        with self.assertRaisesRegex(ValueError,'one-day'):
            period_mask(np.array([cutoff(24)]),np.array([0]),24,26,2)

if __name__=='__main__':unittest.main()
