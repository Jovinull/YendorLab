import hashlib
import tempfile
import unittest
from pathlib import Path

from yendorlab.agents import QAgent, encode_state
from yendorlab.evaluation import run_policy_episode, seed_for, wilson
from yendorlab.training import load_checkpoint, load_policy_checkpoint, save_checkpoint, train, train_episode


BINARY = Path('.local/BrogueCE/bin/brogue')


class TrainingUnitTests(unittest.TestCase):
    def test_q_value_changes_from_experience(self):
        agent = QAgent(seed=1)
        self.assertTrue(agent.update('state', 'h', 1.0, '', True))
        self.assertAlmostEqual(agent.q['state'][0], .2)

    def test_seed_splits_do_not_overlap(self):
        values = {split: {seed_for(123, split, i) for i in range(100)}
                  for split in ('train', 'validation', 'final')}
        self.assertTrue(values['train'].isdisjoint(values['validation']))
        self.assertTrue(values['validation'].isdisjoint(values['final']))

    def test_wilson_denominator(self):
        self.assertIsNone(wilson(0, 0))
        self.assertLess(wilson(0, 5)[1], 1)

    def test_corrupt_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.json'
            save_checkpoint(path, {'checkpoint_version': 999})
            with self.assertRaises(ValueError):
                load_checkpoint(path)


@unittest.skipUnless(BINARY.is_file(), 'native build absent')
class TrainingNativeTests(unittest.TestCase):
    def test_last_valid_truncated_transition_is_updated(self):
        agent = QAgent(seed=3)
        result = train_episode(BINARY.resolve(), agent, seed=123, variant='classic',
                               max_events=1, max_seconds=5, max_no_progress=10,
                               memory_mb=512, alpha=.2, gamma=.95)
        self.assertEqual(result['reason'], 'TRUNCATED_ACTION_LIMIT')
        self.assertEqual(result['updates'], 1)

    def test_train_checkpoint_evaluate_and_resume(self):
        config = {'experiment': 'TEST', 'variant': 'classic', 'episodes': 2,
                  'total_seconds': 20.0, 'episode_seconds': 5.0,
                  'max_events': 50, 'max_no_progress': 15,
                  'memory_mb': 512, 'master_seed': 98123,
                  'epsilon': .5, 'alpha': .2, 'gamma': .95,
                  'validation_every': 2, 'validation_episodes': 1,
                  'training_task': 'exploration_v1'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            first = train(BINARY.resolve(), config, path)
            model = load_checkpoint(path)
            self.assertEqual(first['episodes_total'], 2)
            self.assertGreater(first['updates_total'], 0)
            self.assertGreater(first['changed_updates_total'], 0)
            self.assertTrue(model['q'])
            self.assertEqual(model['q'], load_policy_checkpoint(Path(first['inference_checkpoint'])))
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            seed = seed_for(config['master_seed'], 'validation', 5)
            first_eval = run_policy_episode(BINARY.resolve(), QAgent(model['q'], seed), seed,
                                            max_events=30, max_seconds=5)
            second_eval = run_policy_episode(BINARY.resolve(), QAgent(load_checkpoint(path)['q'], seed), seed,
                                             max_events=30, max_seconds=5)
            self.assertEqual(first_eval['reason'], second_eval['reason'])
            self.assertEqual(first_eval['turn'], second_eval['turn'])
            self.assertEqual(checksum, hashlib.sha256(path.read_bytes()).hexdigest())
            config['episodes'] = 3
            resumed = train(BINARY.resolve(), config, path, resume=True)
            self.assertEqual(resumed['episodes_this_run'], 1)
            self.assertEqual(resumed['episodes_total'], 3)
            self.assertGreater(resumed['updates_total'], first['updates_total'])


if __name__ == '__main__':
    unittest.main()
