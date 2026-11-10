"""Reject silent probability/event misalignment, even with equal row counts."""
import unittest
import numpy as np
from event_component_v3 import validate_event_alignment,validate_training_config
from v3_event_apply import validate_output_path

class EventAlignmentTests(unittest.TestCase):
    def test_strict_checkpoint_selection_accepted(self):
        validate_training_config({'day':25,'full':False,'selection_end_day':26})

    def test_future_checkpoint_selection_rejected(self):
        with self.assertRaisesRegex(ValueError,'selection'):
            validate_training_config({'day':25,'full':False,'selection_end_day':27})

    def test_wrong_training_period_rejected(self):
        with self.assertRaisesRegex(ValueError,'January25'):
            validate_training_config({'day':26,'full':False,'selection_end_day':27})
    def test_missing_output_suffix_rejected(self):
        with self.assertRaisesRegex(ValueError,'suffix'):
            validate_output_path('unused_event_contract_output')

    def test_uppercase_output_suffix_rejected(self):
        with self.assertRaisesRegex(ValueError,'suffix'):
            validate_output_path('unused_event_contract_output.NPY')
    def test_matching_event_order(self):
        validate_event_alignment(np.array([12,18,25]),np.array([12,18,25]),30)

    def test_same_length_permutation_rejected(self):
        with self.assertRaisesRegex(ValueError,'ordering'):
            validate_event_alignment(np.array([12,18,25]),np.array([18,12,25]),30)

    def test_missing_event_rejected(self):
        with self.assertRaises(ValueError):
            validate_event_alignment(np.array([12,18,25]),np.array([12,18]),30)

    def test_row_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'row count'):
            validate_event_alignment(np.array([12,18,25]),np.array([12,18,25]),20)

if __name__=='__main__':unittest.main()
