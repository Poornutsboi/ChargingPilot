from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from chargingpilot.environment.structured_observation import (
    REQUEST_FEATURE_NAMES,
    STATION_FEATURE_NAMES,
    StructuredObservation,
)
from chargingpilot.routing.cache import CACHE_SCHEMA_VERSION
from chargingpilot.routing.models import (
    HierarchicalAction,
    LambdaDecisionContext,
    S1DecisionContext,
    S2DecisionContext,
)
from chargingpilot.trainer.hierarchical_policy import HierarchicalActorCritic


_CHECKPOINT_METADATA_SCHEMA_VERSION = 1
_OBSERVATION_SCHEMA_VERSION = 1


class _FrozenDict(dict):
    def _immutable(self, *_args, **_kwargs):
        raise TypeError("stored feasibility mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo):
        result = type(self)(
            (copy.deepcopy(key, memo), copy.deepcopy(value, memo))
            for key, value in self.items()
        )
        memo[id(self)] = result
        return result


@dataclass(frozen=True)
class HierarchicalPPOConfig:
    rollout_steps: int = 4096
    minibatch_size: int = 256
    update_epochs: int = 4
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    time_unit_minutes: float = 5.0
    clip_range: float = 0.2
    value_coef: float = 0.5
    s1_entropy_coef: float = 0.01
    s2_entropy_coef: float = 0.01
    lambda_entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_popart: bool = False
    popart_beta: float = 0.999
    popart_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("rollout_steps", "minibatch_size", "update_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "time_unit_minutes",
            "clip_range",
            "value_coef",
            "max_grad_norm",
            "popart_epsilon",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gamma", "gae_lambda", "popart_beta"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


HierarchicalPPOTrainerConfig = HierarchicalPPOConfig


@dataclass(frozen=True)
class RolloutTransition:
    observation: StructuredObservation
    action: HierarchicalAction
    s1_context: S1DecisionContext
    s2_context: S2DecisionContext
    lambda_context: LambdaDecisionContext | None
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    elapsed_minutes: float

    def __post_init__(self) -> None:
        values = (
            self.observation,
            self.action,
            self.s1_context,
            self.s2_context,
            self.lambda_context,
        )
        expected = (
            StructuredObservation,
            HierarchicalAction,
            S1DecisionContext,
            S2DecisionContext,
            (LambdaDecisionContext, type(None)),
        )
        if any(not isinstance(value, kind) for value, kind in zip(values, expected)):
            raise TypeError("transition contains an invalid structured field")
        copied = copy.deepcopy(values)
        for name, value in zip(
            ("observation", "action", "s1_context", "s2_context", "lambda_context"),
            copied,
        ):
            object.__setattr__(self, name, value)
        self.observation.request.setflags(write=False)
        self.observation.stations.setflags(write=False)
        self.s1_context.mask.setflags(write=False)
        self.s2_context.mask.setflags(write=False)
        self.s2_context.features.setflags(write=False)
        if self.lambda_context is not None:
            self.lambda_context.mask.setflags(write=False)
            self.lambda_context.features.setflags(write=False)
            self.lambda_context.bins.setflags(write=False)
        request_context = self.s1_context.request_context
        object.__setattr__(
            request_context, "single_routes", _FrozenDict(request_context.single_routes)
        )
        object.__setattr__(
            request_context, "split_routes", _FrozenDict(request_context.split_routes)
        )
        object.__setattr__(
            request_context,
            "arrival_soc_s1",
            _FrozenDict(request_context.arrival_soc_s1),
        )
        for name in ("old_log_prob", "old_value", "reward", "elapsed_minutes"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.elapsed_minutes < 0.0:
            raise ValueError("elapsed_minutes must be nonnegative")
        if type(self.done) is not bool:
            raise TypeError("done must be a bool")


class RolloutBuffer:
    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._transitions: list[RolloutTransition] = []
        self.bootstrap_value = 0.0

    @property
    def transitions(self) -> tuple[RolloutTransition, ...]:
        return tuple(self._transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def append(self, transition: RolloutTransition) -> None:
        if not isinstance(transition, RolloutTransition):
            raise TypeError("rollout accepts only RolloutTransition values")
        if len(self) >= self.capacity:
            raise OverflowError("rollout is already full")
        self._transitions.append(transition)

    def seal(self, bootstrap_value: float) -> None:
        value = float(bootstrap_value)
        if not np.isfinite(value):
            raise ValueError("bootstrap_value must be finite")
        self.bootstrap_value = value

    def collate(self) -> tuple[RolloutTransition, ...]:
        if len(self) != self.capacity:
            raise RuntimeError(
                f"fixed rollout requires {self.capacity} decisions, received {len(self)}"
            )
        return self.transitions


@dataclass(frozen=True)
class PPOLosses:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    s1_entropy: torch.Tensor
    s2_entropy: torch.Tensor
    lambda_entropy: torch.Tensor
    clip_fraction: torch.Tensor


class PopArtValueNormalizer:
    def __init__(self, *, beta: float, epsilon: float) -> None:
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.mean = 0.0
        self.std = 1.0

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def update(self, targets: torch.Tensor, output_layer: nn.Linear) -> None:
        if targets.numel() == 0:
            return
        old_mean, old_std = self.mean, self.std
        target = targets.detach().double()
        batch_mean = float(target.mean().cpu())
        batch_square_mean = float(target.square().mean().cpu())
        old_square_mean = old_std * old_std + old_mean * old_mean
        new_mean = self.beta * old_mean + (1.0 - self.beta) * batch_mean
        new_square_mean = self.beta * old_square_mean + (1.0 - self.beta) * batch_square_mean
        new_std = float(np.sqrt(max(new_square_mean - new_mean * new_mean, self.epsilon)))
        with torch.no_grad():
            output_layer.weight.mul_(old_std / new_std)
            output_layer.bias.mul_(old_std).add_(old_mean).sub_(new_mean).div_(new_std)
        self.mean, self.std = new_mean, new_std

    def state_dict(self) -> dict[str, float]:
        return {"beta": self.beta, "epsilon": self.epsilon, "mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.beta = float(state["beta"])
        self.epsilon = float(state["epsilon"])
        self.mean = float(state["mean"])
        self.std = float(state["std"])


def compute_time_aware_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    elapsed_minutes: np.ndarray,
    bootstrap_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    time_unit_minutes: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.bool_)
    elapsed_minutes = np.asarray(elapsed_minutes, dtype=np.float64)
    if not (rewards.shape == values.shape == dones.shape == elapsed_minutes.shape):
        raise ValueError("rollout arrays must have identical shapes")
    if rewards.ndim != 1:
        raise ValueError("rollout arrays must be one-dimensional")
    if time_unit_minutes <= 0.0 or not np.isfinite(time_unit_minutes):
        raise ValueError("time_unit_minutes must be finite and positive")
    if not (
        np.isfinite(rewards).all()
        and np.isfinite(values).all()
        and np.isfinite(elapsed_minutes).all()
        and np.isfinite(bootstrap_value)
    ):
        raise ValueError("GAE inputs must be finite")
    if bool((elapsed_minutes < 0.0).any()):
        raise ValueError("elapsed_minutes must be nonnegative")

    advantages = np.zeros_like(rewards)
    next_advantage = 0.0
    next_value = float(bootstrap_value)
    for index in range(rewards.size - 1, -1, -1):
        nonterminal = 0.0 if dones[index] else 1.0
        gamma_t = gamma ** (elapsed_minutes[index] / time_unit_minutes)
        delta = rewards[index] + gamma_t * next_value * nonterminal - values[index]
        next_advantage = delta + gamma_t * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


def ppo_probability_ratio(*, new_joint_log_prob: Any, old_joint_log_prob: Any) -> Any:
    if isinstance(new_joint_log_prob, torch.Tensor) or isinstance(old_joint_log_prob, torch.Tensor):
        new = torch.as_tensor(new_joint_log_prob)
        old = torch.as_tensor(old_joint_log_prob, device=new.device, dtype=new.dtype)
        if new.shape != old.shape:
            raise ValueError("new and old joint log probabilities must match")
        return torch.exp(new - old)
    new = np.asarray(new_joint_log_prob)
    old = np.asarray(old_joint_log_prob)
    if new.shape != old.shape:
        raise ValueError("new and old joint log probabilities must match")
    return np.exp(new - old)


def compute_ppo_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    value_targets: torch.Tensor,
    s1_entropy: torch.Tensor,
    s2_entropy: torch.Tensor,
    lambda_entropy: torch.Tensor,
    split_mask: torch.Tensor,
    config: HierarchicalPPOConfig,
) -> PPOLosses:
    ratio = ppo_probability_ratio(
        new_joint_log_prob=new_log_prob, old_joint_log_prob=old_log_prob
    )
    clipped_ratio = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    clipped_values = old_values + (new_values - old_values).clamp(
        -config.clip_range, config.clip_range
    )
    value_loss = 0.5 * torch.maximum(
        (new_values - value_targets).square(),
        (clipped_values - value_targets).square(),
    ).mean()
    mean_s1 = s1_entropy.mean()
    mean_s2 = s2_entropy.mean()
    mean_lambda = (
        lambda_entropy[split_mask].mean()
        if bool(split_mask.any())
        else lambda_entropy.new_zeros(())
    )
    total = (
        policy_loss
        + config.value_coef * value_loss
        - config.s1_entropy_coef * mean_s1
        - config.s2_entropy_coef * mean_s2
        - config.lambda_entropy_coef * mean_lambda
    )
    return PPOLosses(
        total,
        policy_loss,
        value_loss,
        mean_s1,
        mean_s2,
        mean_lambda,
        (torch.abs(ratio - 1.0) > config.clip_range).float().mean(),
    )


class HierarchicalPPOTrainer:
    def __init__(
        self,
        *,
        env: Any,
        config: HierarchicalPPOConfig | None = None,
        policy: nn.Module | None = None,
    ) -> None:
        self.env = env
        self.config = config or HierarchicalPPOConfig()
        self.device = torch.device(self.config.device)
        self._seed_everything(self.config.seed)
        self._minibatch_rng = np.random.default_rng(self.config.seed)
        if policy is None:
            oracle = env.oracle
            station_indices = tuple(
                int(oracle.node_to_index[station_id]) for station_id in oracle.station_ids
            )
            if any(left >= right for left, right in zip(station_indices, station_indices[1:])):
                raise ValueError("oracle station node indices must be strictly ascending")
            policy = HierarchicalActorCritic(
                node_count=len(oracle.node_ids), station_node_indices=station_indices
            )
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        self.value_normalizer = PopArtValueNormalizer(
            beta=self.config.popart_beta, epsilon=self.config.popart_epsilon
        )
        self._current_observation: StructuredObservation | None = None
        self._needs_reset = True
        self._active_rollout_count = 0
        self._completed_rollouts = 0
        self.completed_decisions = 0
        self.completed_updates = 0

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def collect_rollout(self) -> RolloutBuffer:
        buffer = RolloutBuffer(self.config.rollout_steps)
        self._active_rollout_count = 0
        for _ in range(self.config.rollout_steps):
            if self._needs_reset:
                seed = self.config.seed if self.completed_decisions == 0 else None
                self._current_observation, _info = self.env.reset(seed=seed)
                self._needs_reset = False
            observation = self._current_observation
            if not isinstance(observation, StructuredObservation):
                raise TypeError("hierarchical environment must return StructuredObservation")
            s1_context = self.env.s1_context()
            provider = getattr(self.env, "generator", self.env)
            with torch.no_grad():
                sample = self.policy.sample_action(observation, s1_context, provider)
            old_value = self._raw_value(sample.value)
            next_observation, reward, terminated, truncated, info = self.env.step(sample.action)
            done = bool(terminated or truncated)
            transition = RolloutTransition(
                observation=observation,
                action=sample.action,
                s1_context=s1_context,
                s2_context=sample.s2_context,
                lambda_context=sample.lambda_context,
                old_log_prob=float(sample.log_prob.detach().cpu()),
                old_value=old_value,
                reward=float(reward),
                done=done,
                elapsed_minutes=float(info["elapsed_minutes"]),
            )
            buffer.append(transition)
            self._active_rollout_count += 1
            self.completed_decisions += 1
            if done:
                self._current_observation = None
                self._needs_reset = True
            else:
                self._current_observation = next_observation
        bootstrap = 0.0 if buffer.transitions[-1].done else self._value_for_observation(
            self._current_observation
        )
        buffer.seal(bootstrap)
        self._active_rollout_count = 0
        self._completed_rollouts += 1
        return buffer

    def _raw_value(self, value: torch.Tensor) -> float:
        value = value.detach()
        if self.config.use_popart:
            value = self.value_normalizer.denormalize(value)
        return float(value.cpu())

    def _value_for_observation(self, observation: StructuredObservation | None) -> float:
        if observation is None:
            return 0.0
        with torch.no_grad():
            if hasattr(self.policy, "encode"):
                value = self.policy.encode(observation).value.squeeze(0)
            elif hasattr(self.policy, "value"):
                value = self.policy.value(observation)
            elif hasattr(self.policy, "value_bias"):
                value = self.policy.value_bias
            else:
                raise TypeError("policy does not expose value evaluation")
        return self._raw_value(value)

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        transitions = rollout.collate()
        for transition in transitions:
            self._validate_replay_transition(transition)
        rewards = np.array([item.reward for item in transitions], dtype=np.float64)
        old_values_np = np.array([item.old_value for item in transitions], dtype=np.float64)
        dones = np.array([item.done for item in transitions], dtype=np.bool_)
        elapsed = np.array([item.elapsed_minutes for item in transitions], dtype=np.float64)
        advantages_np, returns_np = compute_time_aware_gae(
            rewards=rewards,
            values=old_values_np,
            dones=dones,
            elapsed_minutes=elapsed,
            bootstrap_value=rollout.bootstrap_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            time_unit_minutes=self.config.time_unit_minutes,
        )
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(old_values_np, dtype=torch.float32, device=self.device)
        old_log_prob = torch.tensor(
            [item.old_log_prob for item in transitions], dtype=torch.float32, device=self.device
        )
        if self.config.use_popart:
            self.value_normalizer.update(returns, self._value_output_layer())
            value_targets = self.value_normalizer.normalize(returns)
            old_values_for_loss = self.value_normalizer.normalize(old_values)
        else:
            value_targets = returns
            old_values_for_loss = old_values

        metric_rows: list[dict[str, float]] = []
        indices = np.arange(len(transitions))
        for _epoch in range(self.config.update_epochs):
            self._minibatch_rng.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                selected = indices[start : start + self.config.minibatch_size]
                batch = tuple(transitions[int(index)] for index in selected)
                evaluation = self.policy.evaluate_action(
                    tuple(item.observation for item in batch),
                    tuple(item.action for item in batch),
                    tuple(item.s1_context for item in batch),
                    tuple(item.s2_context for item in batch),
                    tuple(item.lambda_context for item in batch),
                )
                index_tensor = torch.as_tensor(selected, dtype=torch.long, device=self.device)
                split_mask = torch.tensor(
                    [item.lambda_context is not None for item in batch],
                    dtype=torch.bool,
                    device=self.device,
                )
                losses = compute_ppo_loss(
                    new_log_prob=evaluation.log_prob,
                    old_log_prob=old_log_prob[index_tensor],
                    advantages=advantages[index_tensor],
                    new_values=evaluation.value,
                    old_values=old_values_for_loss[index_tensor],
                    value_targets=value_targets[index_tensor],
                    s1_entropy=evaluation.s1_entropy,
                    s2_entropy=evaluation.s2_entropy,
                    lambda_entropy=evaluation.lambda_entropy,
                    split_mask=split_mask,
                    config=self.config,
                )
                if not torch.isfinite(losses.total_loss):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                losses.total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO gradient")
                self.optimizer.step()
                metric_rows.append(
                    {
                        "policy_loss": float(losses.policy_loss.detach().cpu()),
                        "value_loss": float(losses.value_loss.detach().cpu()),
                        "s1_entropy": float(losses.s1_entropy.detach().cpu()),
                        "s2_entropy": float(losses.s2_entropy.detach().cpu()),
                        "lambda_entropy": float(losses.lambda_entropy.detach().cpu()),
                        "clip_fraction": float(losses.clip_fraction.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu()),
                    }
                )
        self.completed_updates += 1
        metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
        metrics.update(
            {
                "mean_reward": float(rewards.mean()),
                "advantage_mean": float(advantages.mean().detach().cpu()),
                "advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
            }
        )
        if self.config.use_popart:
            metrics.update(
                {
                    "popart_mean": self.value_normalizer.mean,
                    "popart_std": self.value_normalizer.std,
                }
            )
        return metrics

    def _value_output_layer(self) -> nn.Linear:
        head = getattr(self.policy, "value_head", None)
        if isinstance(head, nn.Linear):
            return head
        if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
            return head[-1]
        raise TypeError("PopArt requires a linear value output layer")

    @staticmethod
    def _validate_replay_transition(transition: RolloutTransition) -> None:
        action = transition.action
        if not 0 <= action.s1_index < transition.s1_context.mask.size:
            raise ValueError("stored s1 action is out of range")
        if not bool(transition.s1_context.mask[action.s1_index]):
            raise ValueError("stored s1 action is masked")
        if transition.s2_context.s1_index != action.s1_index:
            raise ValueError("stored s2 context does not match s1 action")
        if transition.s2_context.request_context is not transition.s1_context.request_context:
            raise ValueError("stored contexts do not share feasibility context")
        if not 0 <= action.s2_index < transition.s2_context.mask.size:
            raise ValueError("stored s2 action is out of range")
        if not bool(transition.s2_context.mask[action.s2_index]):
            raise ValueError("stored s2 action is masked")
        if action.s2_index == transition.s1_context.mask.size:
            if action.lambda_index is not None or transition.lambda_context is not None:
                raise ValueError("single-stop replay must omit lambda")
            return
        context = transition.lambda_context
        if context is None or action.lambda_index is None:
            raise ValueError("split replay requires lambda context and action")
        if context.request_context is not transition.s1_context.request_context:
            raise ValueError("stored lambda context does not share feasibility context")
        if (context.s1_index, context.s2_index) != (action.s1_index, action.s2_index):
            raise ValueError("stored lambda context does not match action")
        if not 0 <= action.lambda_index < context.mask.size:
            raise ValueError("stored lambda action is out of range")
        if not bool(context.mask[action.lambda_index]):
            raise ValueError("stored lambda action is masked")

    def save_checkpoint(self, path: str | Path, *, resumable: bool = True) -> None:
        if resumable and self._active_rollout_count:
            raise RuntimeError("cannot save a resumable checkpoint with a partial rollout")
        if resumable and self._completed_rollouts == 0:
            raise RuntimeError("resumable checkpoints require a completed rollout boundary")
        checkpoint = {
            "resumable": bool(resumable),
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.config),
            "value_normalizer_state": self.value_normalizer.state_dict(),
            "completed_decisions": self.completed_decisions,
            "completed_updates": self.completed_updates,
            "completed_rollouts": self._completed_rollouts,
            "feature_index_cache_metadata": self._feature_index_cache_metadata(),
        }
        if resumable:
            checkpoint.update(
                {
                    "environment_state": self._environment_state(),
                    "current_observation": copy.deepcopy(self._current_observation),
                    "needs_reset": self._needs_reset,
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_cpu_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                    "minibatch_rng_state": copy.deepcopy(
                        self._minibatch_rng.bit_generator.state
                    ),
                }
            )
        torch.save(checkpoint, Path(path))

    def load_checkpoint(self, path: str | Path, *, resume: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if resume and not checkpoint.get("resumable", False):
            raise ValueError("checkpoint is inference-only and cannot be resumed")
        if checkpoint["trainer_config"] != asdict(self.config):
            raise ValueError("checkpoint trainer configuration does not match")
        active_metadata = self._feature_index_cache_metadata()
        if checkpoint.get("feature_index_cache_metadata") != active_metadata:
            raise ValueError(
                "checkpoint feature/index/cache metadata does not match active trainer"
            )
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.value_normalizer.load_state_dict(checkpoint["value_normalizer_state"])
        self.completed_decisions = int(checkpoint["completed_decisions"])
        self.completed_updates = int(checkpoint["completed_updates"])
        self._completed_rollouts = int(checkpoint["completed_rollouts"])
        if resume:
            self._restore_environment_state(checkpoint["environment_state"])
            self._current_observation = checkpoint["current_observation"]
            self._needs_reset = bool(checkpoint["needs_reset"])
            self._active_rollout_count = 0
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
            if torch.cuda.is_available() and checkpoint["torch_cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
            self._minibatch_rng.bit_generator.state = checkpoint["minibatch_rng_state"]
        return checkpoint

    def _environment_state(self) -> dict[str, Any]:
        factory_state = self._episode_factory_state()
        if hasattr(self.env, "state_dict") and hasattr(self.env, "load_state_dict"):
            return {
                "kind": "state_dict",
                "value": copy.deepcopy(self.env.state_dict()),
                "episode_factory": factory_state,
            }
        excluded = {"episode_factory", "oracle", "generator", "config"}
        state = {
            key: copy.deepcopy(value)
            for key, value in vars(self.env).items()
            if key not in excluded and key != "np_random"
        }
        return {"kind": "attributes", "value": state, "episode_factory": factory_state}

    def _restore_environment_state(self, state: dict[str, Any]) -> None:
        if state["kind"] == "state_dict":
            self.env.load_state_dict(copy.deepcopy(state["value"]))
        else:
            for key, value in state["value"].items():
                setattr(self.env, key, copy.deepcopy(value))
        self._restore_episode_factory_state(state.get("episode_factory"))

    def _episode_factory_state(self) -> dict[str, Any] | None:
        factory = getattr(self.env, "episode_factory", None)
        if factory is None:
            raise RuntimeError(
                "resumable checkpoint requires a restorable episode factory"
            )
        state_dict = getattr(factory, "state_dict", None)
        load_state_dict = getattr(factory, "load_state_dict", None)
        if callable(state_dict) and callable(load_state_dict):
            try:
                state = copy.deepcopy(state_dict())
            except Exception as exc:
                raise RuntimeError("failed to capture episode factory state") from exc
            return {
                "kind": "state_dict",
                "factory_identity": self._episode_factory_identity(factory),
                "value": state,
            }
        if callable(state_dict) != callable(load_state_dict):
            raise RuntimeError(
                "episode factory must implement both state_dict and load_state_dict"
            )
        if all(hasattr(factory, name) for name in ("_cursor", "_order", "_rng")):
            cursor = factory._cursor
            order = factory._order
            rng = factory._rng
            if (
                type(cursor) is not int
                or not isinstance(order, (list, tuple))
                or any(type(index) is not int for index in order)
                or cursor < 0
                or cursor > len(order)
                or not callable(getattr(rng, "getstate", None))
                or not callable(getattr(rng, "setstate", None))
                or not hasattr(factory, "__dict__")
            ):
                raise RuntimeError(
                    "episode factory cursor/order/RNG state is not completely restorable"
                )
            return {
                "kind": "manifest_cursor",
                "factory_identity": self._episode_factory_identity(factory),
                "cursor": cursor,
                "order": copy.deepcopy(order),
                "rng_state": copy.deepcopy(rng.getstate()),
            }
        raise RuntimeError(
            "resumable checkpoint cannot capture complete episode factory state"
        )

    def _restore_episode_factory_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            raise ValueError("resumable checkpoint is missing episode factory state")
        factory = self.env.episode_factory
        if state.get("factory_identity") != self._episode_factory_identity(factory):
            raise ValueError("checkpoint episode factory identity does not match")
        if state["kind"] == "state_dict":
            factory.load_state_dict(copy.deepcopy(state["value"]))
            return
        if state["kind"] != "manifest_cursor":
            raise ValueError("checkpoint episode factory state kind is unsupported")
        factory._cursor = int(state["cursor"])
        factory._order = copy.deepcopy(state["order"])
        factory._rng.setstate(state["rng_state"])

    @staticmethod
    def _episode_factory_identity(factory: Any) -> dict[str, Any]:
        return {
            "type": f"{type(factory).__module__}.{type(factory).__qualname__}",
            "config": repr(getattr(factory, "config", None)),
            "episode_paths": tuple(
                str(Path(path).resolve())
                for path in getattr(factory, "_episode_paths", ())
            ),
        }

    def _feature_index_cache_metadata(self) -> dict[str, Any]:
        oracle = getattr(self.env, "oracle", None)
        return {
            "metadata_schema_version": _CHECKPOINT_METADATA_SCHEMA_VERSION,
            "observation_schema": {
                "schema_version": _OBSERVATION_SCHEMA_VERSION,
                "request_feature_names": tuple(REQUEST_FEATURE_NAMES),
                "station_feature_names": tuple(STATION_FEATURE_NAMES),
                "request_shape": (len(REQUEST_FEATURE_NAMES),),
                "station_shape": (72, len(STATION_FEATURE_NAMES)),
            },
            "node_ids": tuple(getattr(oracle, "node_ids", ())),
            "station_ids": tuple(getattr(oracle, "station_ids", ())),
            "node_to_index": tuple(
                sorted(
                    (int(node_id), int(index))
                    for node_id, index in getattr(oracle, "node_to_index", {}).items()
                )
            ),
            "station_node_indices": tuple(
                int(value)
                for value in getattr(self.policy, "station_node_indices", torch.tensor([])).tolist()
            ),
            "route_cache_identity": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key": copy.deepcopy(getattr(oracle, "_cache_key", None)),
                "directed": copy.deepcopy(getattr(oracle, "directed", None)),
                "path": None
                if getattr(oracle, "cache_path", None) is None
                else str(Path(oracle.cache_path).resolve()),
            },
        }


__all__ = [
    "HierarchicalPPOConfig",
    "HierarchicalPPOTrainerConfig",
    "HierarchicalPPOTrainer",
    "PPOLosses",
    "PopArtValueNormalizer",
    "RolloutBuffer",
    "RolloutTransition",
    "compute_ppo_loss",
    "compute_time_aware_gae",
    "ppo_probability_ratio",
]
