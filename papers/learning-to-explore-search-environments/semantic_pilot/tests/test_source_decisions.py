"""Verify source-only utility extraction and a nontrivial exact certificate."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_source_decisions import source_model
from searchprobe.decisions import audit_decision_model


class SourceDecisionTests(unittest.TestCase):
    def records(self):
        return [{'name': corpus + '_bm25', 'corpus_name': corpus, 'backend': 'bm25',
                 'tasks': {'train': {'ids': [str(i) for i in range(128)],
                                     'scores': [row for _ in range(128)]},
                           'test': {'poison': 'must never be read'}}}
                for corpus, row in [('scifact', [0, 0, 0, 0, 1]),
                                    ('fiqa', [1, 0, 0, 0, 0]),
                                    ('nfcorpus', [0, 1, 0, 0, 0])]]

    def test_target_data_and_other_source_partitions_cannot_change_model(self):
        records = self.records()
        before = source_model(records, 'scifact', 'bm25')
        changed = copy.deepcopy(records)
        changed[0]['tasks'] = {'poison': 'no target access'}
        for record in changed[1:]:
            record['tasks']['test'] = {'different': True}
            record['tasks']['utility'] = {'different': True}
        self.assertEqual(before, source_model(changed, 'scifact', 'bm25'))
        report = audit_decision_model(before)
        self.assertEqual(report['decision_radius']['exact'], '1/2')
        self.assertEqual(report['certificate']['duality_gap']['exact'], '0')
        self.assertFalse(report['target_regret_guarantee'])

    def test_missing_world_duplicate_ids_and_invalid_scores_fail(self):
        records = self.records()
        with self.assertRaisesRegex(ValueError, 'other two'):
            source_model(records[:-1], 'scifact', 'bm25')
        records[1]['tasks']['train']['ids'][1] = '0'
        with self.assertRaisesRegex(ValueError, '128 unique'):
            source_model(records, 'scifact', 'bm25')
        records = self.records()
        records[1]['tasks']['train']['scores'][0] = [2, 0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'bounded'):
            source_model(records, 'scifact', 'bm25')


if __name__ == '__main__':
    unittest.main()
