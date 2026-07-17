import random
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from chargingpilot.routing.models import HierarchicalAction
from tests.test_hierarchical_policy import _Provider, _contexts, _observation
from chargingpilot.trainer.hierarchical_policy import PolicyEvaluation, PolicySample
from chargingpilot.trainer.hierarchical_ppo_trainer import (
    HierarchicalPPOConfig,
    HierarchicalPPOTrainer,
    RolloutBuffer,
    RolloutTransition,
    compute_time_aware_gae,
    compute_ppo_loss,
    ppo_probability_ratio,
)


class TimeAwareGAETests(unittest.TestCase):
    def test_exact_config_defaults(self):
        config = HierarchicalPPOConfig()

        self.assertEqual(config.rollout_steps, 4096)
        self.assertEqual(config.minibatch_size, 256)
        self.assertEqual(config.update_epochs, 4)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.clip_range, 0.2)
        self.assertEqual(config.gae_lambda, 0.95)
        self.assertEqual(config.value_coef, 0.5)
        self.assertEqual(config.max_grad_norm, 0.5)
        self.assertEqual(
            (config.s1_entropy_coef, config.s2_entropy_coef, config.lambda_entropy_coef),
            (0.01, 0.01, 0.005),
        )

    def test_exact_time_aware_gae_and_terminal_reset(self):
        config = HierarchicalPPOConfig(gamma=0.99, gae_lambda=0.95)
        advantages, returns = compute_time_aware_gae(
            rewards=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            values=np.array([0.5, 0.25, -0.5], dtype=np.float64),
            dones=np.array([False, True, False]),
            elapsed_minutes=np.array([5.0, 10.0, 2.5], dtype=np.float64),
            bootstrap_value=0.75,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            time_unit_minutes=config.time_unit_minutes,
        )

        gamma0 = 0.99
        gamma1 = 0.99**2
        gamma2 = 0.99**0.5
        expected_advantage_1 = 2.0 - 0.25
        expected_advantage_0 = (
            1.0 + gamma0 * 0.25 - 0.5
            + gamma0 * 0.95 * expected_advantage_1
        )
        expected_advantage_2 = 3.0 + gamma2 * 0.75 - (-0.5)
        expected = np.array(
            [expected_advantage_0, expected_advantage_1, expected_advantage_2]
        )
        np.testing.assert_allclose(advantages, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            returns,
            expected + np.array([0.5, 0.25, -0.5]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_joint_probability_ratio_uses_summed_conditional_log_probability(self):
        ratio = ppo_probability_ratio(
            new_joint_log_prob=np.array([0.1 + 0.2 + 0.3, -0.4 + 0.2]),
            old_joint_log_prob=np.array([0.0 + 0.1 + 0.2, -0.2 + 0.1]),
        )
        np.testing.assert_allclose(ratio, np.exp(np.array([0.3, -0.1])))

    def test_clipped_policy_value_and_separate_entropy_loss(self):
        losses = compute_ppo_loss(
            new_log_prob=torch.log(torch.tensor([1.5, 0.5])),
            old_log_prob=torch.zeros(2),
            advantages=torch.tensor([1.0, -1.0]),
            new_values=torch.tensor([3.0, -2.0]),
            old_values=torch.tensor([1.0, -1.0]),
            value_targets=torch.tensor([2.0, 1.0]),
            s1_entropy=torch.tensor([2.0, 4.0]),
            s2_entropy=torch.tensor([1.0, 3.0]),
            lambda_entropy=torch.tensor([7.0, 0.0]),
            split_mask=torch.tensor([True, False]),
            config=HierarchicalPPOConfig(),
        )

        self.assertAlmostEqual(float(losses.policy_loss), -0.2, places=6)
        self.assertAlmostEqual(float(losses.value_loss), 2.5, places=6)
        self.assertAlmostEqual(float(losses.s1_entropy), 3.0)
        self.assertAlmostEqual(float(losses.s2_entropy), 2.0)
        self.assertAlmostEqual(float(losses.lambda_entropy), 7.0)
        self.assertAlmostEqual(float(losses.total_loss), -0.2 + 1.25 - 0.03 - 0.02 - 0.035, places=6)

    def test_lambda_entropy_is_exactly_zero_without_split_samples(self):
        losses = compute_ppo_loss(
            new_log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            advantages=torch.zeros(2),
            new_values=torch.zeros(2),
            old_values=torch.zeros(2),
            value_targets=torch.zeros(2),
            s1_entropy=torch.zeros(2),
            s2_entropy=torch.zeros(2),
            lambda_entropy=torch.tensor([9.0, 8.0]),
            split_mask=torch.zeros(2, dtype=torch.bool),
            config=HierarchicalPPOConfig(),
        )

        self.assertEqual(float(losses.lambda_entropy), 0.0)


def _transition(index: int = 0) -> RolloutTransition:
    s1_context, s2_context, lambda_context = _contexts(none=True)
    return RolloutTransition(
        observation=_observation(request_shift=float(index)),
        action=HierarchicalAction(3, 72, None),
        s1_context=s1_context,
        s2_context=s2_context,
        lambda_context=lambda_context,
        old_log_prob=-0.25,
        old_value=0.5,
        reward=float(index + 1),
        done=False,
        elapsed_minutes=5.0,
    )


class RolloutStorageTests(unittest.TestCase):
    def test_transition_and_context_arrays_are_immutable(self):
        transition = _transition()

        with self.assertRaises(FrozenInstanceError):
            transition.reward = 3.0
        with self.assertRaises(ValueError):
            transition.observation.request[0] = 99.0
        with self.assertRaises(ValueError):
            transition.s1_context.mask[3] = False

    def test_buffer_only_collates_at_exact_fixed_capacity(self):
        buffer = RolloutBuffer(capacity=2)
        buffer.append(_transition(0))

        with self.assertRaises(RuntimeError):
            buffer.collate()

        buffer.append(_transition(1))
        batch = buffer.collate()
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch, buffer.transitions)
        with self.assertRaises(OverflowError):
            buffer.append(_transition(2))


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.2))
        self.value_bias = nn.Parameter(torch.tensor(0.1))

    def sample_action(self, observation, s1_context, provider, *, deterministic=False):
        s1_index = int(np.flatnonzero(s1_context.mask)[0])
        s2_context = provider.build_s2_context(s1_context.request_context, s1_index)
        s2_index = int(np.flatnonzero(s2_context.mask)[0])
        lambda_context = None
        lambda_index = None
        if s2_index != 72:
            lambda_context = provider.build_lambda_context(
                s1_context.request_context, s1_index, s2_index
            )
            lambda_index = int(np.flatnonzero(lambda_context.mask)[0])
        value = self.value_bias
        return PolicySample(
            log_prob=self.logit,
            value=value,
            s1_entropy=self.logit.square() + 0.1,
            s2_entropy=self.logit.square() + 0.2,
            lambda_entropy=self.logit.square() + 0.3 if lambda_context else self.logit * 0.0,
            action=HierarchicalAction(s1_index, s2_index, lambda_index),
            s2_context=s2_context,
            lambda_context=lambda_context,
        )

    def evaluate_action(self, observation, action, s1_context, s2_context, lambda_context):
        batch = len(action) if isinstance(action, tuple) else 1
        log_prob = self.logit.expand(batch)
        value = self.value_bias.expand(batch)
        s1_entropy = (self.logit.square() + 0.1).expand(batch)
        s2_entropy = (self.logit.square() + 0.2).expand(batch)
        lambda_entropy = torch.stack(
            [self.logit.square() + 0.3 if item is not None else self.logit * 0.0 for item in (lambda_context if isinstance(lambda_context, tuple) else (lambda_context,))]
        )
        return PolicyEvaluation(log_prob, value, s1_entropy, s2_entropy, lambda_entropy)


class _StatefulFactory:
    def __init__(self):
        self.cursor = 0

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])


class _ProviderEnv:
    def __init__(self, episode_length=3, *, cache_key="cache-A", episode_factory=None):
        self.episode_length = episode_length
        self.position = 0
        self.reset_count = 0
        self.generator = None
        self.episode_factory = episode_factory or _StatefulFactory()
        self.oracle = type(
            "Oracle",
            (),
            {
                "node_ids": tuple(range(100)),
                "station_ids": tuple(range(10, 82)),
                "node_to_index": {node: node for node in range(100)},
            },
        )()
        self.oracle._cache_key = cache_key
        self.generator = self

    def reset(self, *, seed=None):
        self.position = 0
        self.reset_count += 1
        return _observation(request_shift=0.0), {}

    def s1_context(self):
        return _contexts(none=True)[0]

    def build_s2_context(self, request_context, s1_index):
        return _contexts(none=True, request_context=request_context)[1]

    def build_lambda_context(self, request_context, s1_index, s2_index):
        return None

    def step(self, action):
        self.position += 1
        done = self.position == self.episode_length
        return (
            _observation(request_shift=float(self.position)),
            float(self.position),
            done,
            False,
            {"elapsed_minutes": float(self.position)},
        )

    def state_dict(self):
        return {"position": self.position, "reset_count": self.reset_count}

    def load_state_dict(self, state):
        self.position = int(state["position"])
        self.reset_count = int(state["reset_count"])


class HierarchicalPPOTrainerTests(unittest.TestCase):
    def test_legacy_trainer_modules_reexport_hierarchical_trainer(self):
        from chargingpilot.trainer import HierarchicalPPOConfig as PackageConfig
        from chargingpilot.trainer import HierarchicalPPOTrainer as PackageTrainer
        from chargingpilot.trainer.ppo_trainer import HierarchicalPPOConfig as LegacyConfig
        from chargingpilot.trainer.ppo_trainer import HierarchicalPPOTrainer as LegacyTrainer

        self.assertIs(PackageTrainer, HierarchicalPPOTrainer)
        self.assertIs(LegacyTrainer, HierarchicalPPOTrainer)
        self.assertIs(PackageConfig, HierarchicalPPOConfig)
        self.assertIs(LegacyConfig, HierarchicalPPOConfig)

    def test_policy_construction_uses_explicit_ascending_oracle_station_indices(self):
        env = _ProviderEnv()
        trainer = HierarchicalPPOTrainer(
            env=env, config=HierarchicalPPOConfig(rollout_steps=1, device="cpu")
        )

        self.assertEqual(
            tuple(trainer.policy.station_node_indices.tolist()), tuple(range(10, 82))
        )

    def test_fixed_rollout_continues_mid_episode_and_crosses_episode_boundary(self):
        env = _ProviderEnv(episode_length=3)
        trainer = HierarchicalPPOTrainer(
            env=env,
            policy=_TinyPolicy(),
            config=HierarchicalPPOConfig(rollout_steps=2, minibatch_size=2, update_epochs=1, device="cpu"),
        )

        first = trainer.collect_rollout()
        self.assertEqual(env.position, 2)
        self.assertEqual(env.reset_count, 1)
        second = trainer.collect_rollout()

        self.assertEqual([item.done for item in first.transitions], [False, False])
        self.assertEqual([item.done for item in second.transitions], [True, False])
        self.assertEqual(env.position, 1)
        self.assertEqual(env.reset_count, 2)

    def test_update_normalizes_advantages_and_has_finite_gradients(self):
        env = _ProviderEnv(episode_length=5)
        policy = _TinyPolicy()
        trainer = HierarchicalPPOTrainer(
            env=env,
            policy=policy,
            config=HierarchicalPPOConfig(rollout_steps=2, minibatch_size=2, update_epochs=2, device="cpu"),
        )
        rollout = trainer.collect_rollout()
        before = [parameter.detach().clone() for parameter in policy.parameters()]

        metrics = trainer.update(rollout)

        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertAlmostEqual(metrics["advantage_mean"], 0.0, places=6)
        self.assertAlmostEqual(metrics["advantage_std"], 1.0, places=6)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, policy.parameters())))
        for parameter in policy.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_replay_rejects_action_that_does_not_match_stored_mask(self):
        env = _ProviderEnv()
        trainer = HierarchicalPPOTrainer(
            env=env,
            policy=_TinyPolicy(),
            config=HierarchicalPPOConfig(rollout_steps=1, minibatch_size=1, update_epochs=1, device="cpu"),
        )
        transition = _transition()
        invalid = replace(transition, action=HierarchicalAction(4, 72, None))
        buffer = RolloutBuffer(capacity=1)
        buffer.append(invalid)

        with self.assertRaisesRegex(ValueError, "masked"):
            trainer.update(buffer)

    def test_checkpoint_roundtrip_restores_rng_and_next_transition(self):
        config = HierarchicalPPOConfig(
            rollout_steps=2, minibatch_size=2, update_epochs=1, seed=17, device="cpu"
        )
        control = HierarchicalPPOTrainer(env=_ProviderEnv(episode_length=5), policy=_TinyPolicy(), config=config)
        control.collect_rollout()
        control.env.episode_factory.cursor = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            control.save_checkpoint(path)
            expected = control.collect_rollout().transitions[0]
            expected_random = (random.random(), np.random.random(), torch.rand(()).item())

            resumed = HierarchicalPPOTrainer(env=_ProviderEnv(episode_length=5), policy=_TinyPolicy(), config=config)
            resumed.load_checkpoint(path)
            self.assertEqual(resumed.env.episode_factory.cursor, 4)
            actual = resumed.collect_rollout().transitions[0]
            actual_random = (random.random(), np.random.random(), torch.rand(()).item())

        self.assertEqual(actual.action, expected.action)
        self.assertEqual(actual.reward, expected.reward)
        self.assertEqual(actual.elapsed_minutes, expected.elapsed_minutes)
        np.testing.assert_array_equal(actual.observation.request, expected.observation.request)
        np.testing.assert_allclose(actual_random, expected_random, rtol=0.0, atol=0.0)

    def test_checkpoint_load_skips_incompatible_cuda_rng_state(self):
        config = HierarchicalPPOConfig(
            rollout_steps=1, minibatch_size=1, update_epochs=1, seed=17, device="cpu"
        )
        source = HierarchicalPPOTrainer(
            env=_ProviderEnv(), policy=_TinyPolicy(), config=config
        )
        source.collect_rollout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            source.save_checkpoint(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["torch_cuda_rng_state"] = object()
            torch.save(checkpoint, path)

            resumed = HierarchicalPPOTrainer(
                env=_ProviderEnv(), policy=_TinyPolicy(), config=config
            )
            with patch(
                "chargingpilot.trainer.hierarchical_ppo_trainer.torch.cuda.is_available",
                return_value=True,
            ), patch(
                "chargingpilot.trainer.hierarchical_ppo_trainer.torch.cuda.set_rng_state_all",
                side_effect=TypeError("RNG state must be a torch.ByteTensor"),
            ):
                resumed.load_checkpoint(path)

        self.assertEqual(resumed.completed_decisions, 1)

    def test_resumable_checkpoint_rejects_stateless_episode_factory(self):
        trainer = HierarchicalPPOTrainer(
            env=_ProviderEnv(episode_factory=lambda: None),
            policy=_TinyPolicy(),
            config=HierarchicalPPOConfig(rollout_steps=1, device="cpu"),
        )
        trainer.collect_rollout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-factory.pt"
            with self.assertRaisesRegex(RuntimeError, "episode factory"):
                trainer.save_checkpoint(path)
            self.assertFalse(path.exists())

    def test_checkpoint_metadata_mismatch_rejects_before_any_state_mutation(self):
        config = HierarchicalPPOConfig(rollout_steps=1, seed=23, device="cpu")
        source = HierarchicalPPOTrainer(
            env=_ProviderEnv(cache_key="cache-A"), policy=_TinyPolicy(), config=config
        )
        source.collect_rollout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.pt"
            source.save_checkpoint(path)
            target = HierarchicalPPOTrainer(
                env=_ProviderEnv(cache_key="cache-B"), policy=_TinyPolicy(), config=config
            )
            with torch.no_grad():
                target.policy.logit.fill_(9.0)
            target.completed_decisions = 99
            before_model = {
                key: value.detach().clone()
                for key, value in target.policy.state_dict().items()
            }
            before_optimizer = target.optimizer.state_dict()
            before_python_rng = random.getstate()
            before_numpy_rng = np.random.get_state()
            before_torch_rng = torch.get_rng_state().clone()
            before_env = target.env.state_dict()

            with self.assertRaisesRegex(ValueError, "metadata"):
                target.load_checkpoint(path)

        self.assertEqual(target.completed_decisions, 99)
        for key, value in before_model.items():
            torch.testing.assert_close(target.policy.state_dict()[key], value)
        self.assertEqual(target.optimizer.state_dict(), before_optimizer)
        self.assertEqual(random.getstate(), before_python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], before_numpy_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), before_torch_rng)
        self.assertEqual(target.env.state_dict(), before_env)

    def test_resumable_checkpoint_rejects_partial_rollout(self):
        trainer = HierarchicalPPOTrainer(
            env=_ProviderEnv(),
            policy=_TinyPolicy(),
            config=HierarchicalPPOConfig(rollout_steps=2, device="cpu"),
        )
        trainer._active_rollout_count = 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "partial rollout"):
                trainer.save_checkpoint(Path(directory) / "bad.pt")


if __name__ == "__main__":
    unittest.main()
