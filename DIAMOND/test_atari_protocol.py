import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch
from atari_protocol import conditions, metrics, summarize, wilson

class ProtocolTests(unittest.TestCase):
    def test_shared_storage_git_failure_preserves_source_provenance(self):
        from diamond_atari_runner import source_provenance
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'src').mkdir()
            (root/'src'/'agent.py').write_text('x=1\n')
            failure = subprocess.CompletedProcess([], 128, '', 'dubious ownership')
            with patch('diamond_atari_runner.subprocess.run', return_value=failure):
                result = source_provenance(root)
            self.assertIsNone(result['git_commit'])
            self.assertEqual(result['git_metadata_warning'], 'dubious ownership')
            self.assertEqual(len(result['source_sha256']['src/agent.py']), 64)

    def test_last_budget_step_counts(self):
        self.assertEqual(metrics([0]*99+[1], 1)['first_success_step'], 100)
        with self.assertRaises(ValueError): metrics([0]*100+[1], 1)

    def test_raw_score_is_not_event_count(self):
        m = metrics([5, -3, 0, 1], 5)
        self.assertEqual(m['positive_steps'], 2)
        self.assertEqual(m['clipped_return'], 1)
        self.assertEqual(m['raw_return'], 3)
        self.assertFalse(m['success'])

    def test_early_end_has_fixed_budget_no_extra_episode(self):
        m = metrics([0, 1], 1)
        self.assertEqual((m['executed_steps'], m['budget']), (2, 100))

    def test_conditions(self):
        self.assertEqual(len(conditions()), 12)
        self.assertEqual(conditions()[:2], [('normal', 0.), ('sham', 0.)])
        self.assertTrue(all(0 <= x <= 1 for _, x in conditions()))

    def test_paired_drop_and_missing_trials(self):
        rows = [dict(game='Pong', domain='ale', condition=c, level=0., case=i,
                     success=s, clipped_return=0) for c, vals in [('normal', [1,0]), ('sham',[1,0])]
                for i,s in enumerate(vals)]
        self.assertEqual(summarize(rows, 2)[1]['drop_vs_normal'], 0)
        with self.assertRaises(ValueError): summarize(rows[:-1], 2)
        with self.assertRaises(ValueError): summarize(rows+rows[:1], 2)

    def test_uncertainty_not_zero_when_no_success(self):
        lo, hi = wilson(0,20)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, .1)

if __name__ == '__main__': unittest.main()
