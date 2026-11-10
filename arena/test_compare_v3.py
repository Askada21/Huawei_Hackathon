import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_v3 import assert_valid_artifact, checked_predictions, auc_or_none


class ComparisonContracts(unittest.TestCase):
    def test_rejects_known_invalid_stacking_paths(self):
        for path in ['arena_runs/calibrated_converged/valid_pred.npy',
                     'arena_runs/example/valid_calibrated.npy']:
            with self.assertRaises(ValueError):
                assert_valid_artifact(path)

    def test_rejects_mismatched_event_order_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'valid_pred.npy'
            np.save(path, np.full((2, 10), .1))
            np.save(path.parent / 'valid_events.npy', np.array([7, 6]))
            with self.assertRaises(ValueError):
                checked_predictions(path, np.array([6, 7]))
            np.save(path.parent / 'valid_events.npy', np.array([6, 7]))
            np.save(path, np.full((2, 10), np.nan))
            with self.assertRaises(ValueError):
                checked_predictions(path, np.array([6, 7]))

    def test_single_class_subgroups_have_no_auc(self):
        self.assertIsNone(auc_or_none(np.zeros(5), np.ones(5) * .1))
        self.assertIsNone(auc_or_none(np.array([]), np.array([])))


if __name__ == '__main__':
    unittest.main()
