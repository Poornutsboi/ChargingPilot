from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from chargingpilot.trainer.hierarchical_ppo_trainer import (
    HierarchicalPPOConfig,
    HierarchicalPPOTrainer,
    HierarchicalPPOTrainerConfig,
)

from chargingpilot.environment.split_charging_env import SplitChargingRequestEnv


@dataclass(frozen=True)
class PPOTrainerConfig:
    total_updates: int = 100
    episodes_per_update: int = 8
    batch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_popart: bool = False
    popart_beta: float = 0.999
    popart_epsilon: float = 1.0e-5


@dataclass
class RolloutTransition:
    episode_index: int
    decision_id: int
    obs: np.ndarray
    action: int
    old_log_prob: float
    value: float
    mask: np.ndarray
    vehicle_id: int
    reward: float = 0.0
    next_obs: np.ndarray | None = None
    next_vehicle_id: int | None = None
    next_value: float = 0.0
    done: bool = False
    finalized: bool = False


class MaskedActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        use_popart: bool = False,
        popart_beta: float = 0.999,
        popart_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.use_popart = bool(use_popart)
        self.popart_beta = float(popart_beta)
        self.popart_epsilon = float(popart_epsilon)
        self.shared = nn.Sequential(
            nn.Linear(int(obs_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(int(hidden_dim), int(action_dim))
        self.value_head = nn.Linear(int(hidden_dim), 1)
        self.register_buffer("popart_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("popart_std", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_raw_value(obs)

    def forward_normalized_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def forward_raw_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, normalized_values = self.forward_normalized_value(obs)
        return logits, self.denormalize_value(normalized_values)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, values = self.forward(obs)
        masked_logits = logits.masked_fill(~mask.bool(), -1.0e9)
        return Categorical(logits=masked_logits), values

    def distribution_with_normalized_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        logits, values = self.forward_normalized_value(obs)
        masked_logits = logits.masked_fill(~mask.bool(), -1.0e9)
        return Categorical(logits=masked_logits), values

    def normalize_value(self, values: torch.Tensor) -> torch.Tensor:
        if not self.use_popart:
            return values
        mean = self.popart_mean.to(device=values.device, dtype=values.dtype)
        std = self.popart_std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize_value(self, values: torch.Tensor) -> torch.Tensor:
        if not self.use_popart:
            return values
        mean = self.popart_mean.to(device=values.device, dtype=values.dtype)
        std = self.popart_std.to(device=values.device, dtype=values.dtype)
        return values * std + mean

    def update_popart(self, targets: torch.Tensor) -> None:
        if not self.use_popart or targets.numel() == 0:
            return
        target_values = targets.detach().to(
            device=self.popart_mean.device,
            dtype=self.popart_mean.dtype,
        )
        old_mean = self.popart_mean.detach().clone()
        old_std = self.popart_std.detach().clone()
        beta = float(self.popart_beta)
        batch_mean = target_values.mean()
        batch_square_mean = torch.square(target_values).mean()
        old_square_mean = torch.square(old_std) + torch.square(old_mean)
        new_mean = beta * old_mean + (1.0 - beta) * batch_mean
        new_square_mean = beta * old_square_mean + (1.0 - beta) * batch_square_mean
        new_var = torch.clamp(
            new_square_mean - torch.square(new_mean),
            min=float(self.popart_epsilon),
        )
        new_std = torch.sqrt(new_var)

        with torch.no_grad():
            scale = old_std / new_std
            self.value_head.weight.mul_(scale)
            self.value_head.bias.mul_(old_std).add_(old_mean).sub_(new_mean).div_(new_std)
            self.popart_mean.copy_(new_mean)
            self.popart_std.copy_(new_std)

    def popart_metrics(self) -> dict[str, float]:
        return {
            "popart_mean": float(self.popart_mean.detach().cpu().item()),
            "popart_std": float(self.popart_std.detach().cpu().item()),
        }


class PPOTrainer:
    def __init__(
        self,
        *,
        env: SplitChargingRequestEnv,
        config: PPOTrainerConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or PPOTrainerConfig()
        self.device = torch.device(self.config.device)
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(self.env.action_space.n)
        self.policy = MaskedActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=int(self.config.hidden_dim),
            use_popart=bool(self.config.use_popart),
            popart_beta=float(self.config.popart_beta),
            popart_epsilon=float(self.config.popart_epsilon),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.config.learning_rate),
        )

    def train(
        self,
        on_update: Callable[[dict[str, float], int], None] | None = None,
    ) -> dict[str, float]:
        latest_metrics: dict[str, float] = {}
        for update in range(1, int(self.config.total_updates) + 1):
            transitions = self._collect_rollouts()
            latest_metrics = self._update_policy(transitions)
            latest_metrics["updates"] = int(update)
            latest_metrics["rollout_size"] = int(len(transitions))
            if on_update is not None:
                on_update(dict(latest_metrics), int(update))
        return latest_metrics

    def _collect_rollouts(self) -> list[RolloutTransition]:
        transitions: list[RolloutTransition] = []
        for episode_index in range(int(self.config.episodes_per_update)):
            index_by_decision: dict[int, int] = {}
            obs, info = self.env.reset()
            self._apply_finalized(transitions, index_by_decision, info)
            done = False
            while not done:
                mask = self.env.action_masks()
                vehicle_id = self.env.current_vehicle_id
                if vehicle_id is None:
                    break
                action, log_prob, value = self._sample_action(obs, mask)
                next_obs, _reward, done, _truncated, info = self.env.step(action)
                accepted = info.get("accepted_decision", {})
                decision_id = int(accepted.get("decision_id", len(index_by_decision) + 1))
                transition = RolloutTransition(
                    episode_index=int(episode_index),
                    decision_id=int(decision_id),
                    obs=np.asarray(obs, dtype=np.float32),
                    action=int(action),
                    old_log_prob=float(log_prob),
                    value=float(value),
                    mask=np.asarray(mask, dtype=bool),
                    vehicle_id=int(vehicle_id),
                )
                index_by_decision[int(decision_id)] = len(transitions)
                transitions.append(transition)
                self._apply_finalized(transitions, index_by_decision, info)
                obs = next_obs

        unfinished = [item.vehicle_id for item in transitions if not item.finalized]
        if unfinished:
            raise RuntimeError(f"Missing lazy rewards for vehicles: {unfinished[:10]}")
        return transitions

    def _sample_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy.distribution(obs_tensor, mask_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _apply_finalized(
        self,
        transitions: list[RolloutTransition],
        index_by_decision: dict[int, int],
        info: dict,
    ) -> None:
        for item in info.get("finalized_transitions", []):
            decision_id = int(item["decision_id"])
            index = index_by_decision.get(decision_id)
            if index is None:
                continue
            transitions[index].reward = float(item["reward"])
            next_obs = item.get("next_observation")
            if next_obs is not None:
                transitions[index].next_obs = np.asarray(next_obs, dtype=np.float32)
                transitions[index].next_value = float(
                    self._value_for_observation(transitions[index].next_obs)
                )
            transitions[index].next_vehicle_id = (
                None if item.get("next_vehicle_id") is None else int(item["next_vehicle_id"])
            )
            transitions[index].done = bool(item.get("done", False))
            transitions[index].finalized = True

    def _value_for_observation(self, obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _logits, value = self.policy.forward(obs_tensor)
        return float(value.squeeze(0).item())

    def _update_policy(self, transitions: list[RolloutTransition]) -> dict[str, float]:
        if not transitions:
            metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
            }
            if self.policy.use_popart:
                metrics.update(self.policy.popart_metrics())
            return metrics
        obs = torch.as_tensor(
            np.stack([item.obs for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in transitions],
            dtype=torch.long,
            device=self.device,
        )
        masks = torch.as_tensor(
            np.stack([item.mask for item in transitions]),
            dtype=torch.bool,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            [item.old_log_prob for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.as_tensor(
            [item.value for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        next_values = torch.as_tensor(
            [item.next_value for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        next_indices = self._build_next_indices(transitions)
        advantages, returns = self._compute_gae(rewards, values, next_values, dones, next_indices)
        value_targets = returns
        if self.policy.use_popart:
            self.policy.update_popart(returns)
            value_targets = self.policy.normalize_value(returns)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

        indices = np.arange(len(transitions))
        latest = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }
        if self.policy.use_popart:
            latest.update(self.policy.popart_metrics())
        for _epoch in range(int(self.config.update_epochs)):
            np.random.shuffle(indices)
            for start in range(0, len(indices), int(self.config.batch_size)):
                batch_index = torch.as_tensor(
                    indices[start : start + int(self.config.batch_size)],
                    dtype=torch.long,
                    device=self.device,
                )
                if self.policy.use_popart:
                    dist, new_values = self.policy.distribution_with_normalized_value(
                        obs[batch_index],
                        masks[batch_index],
                    )
                else:
                    dist, new_values = self.policy.distribution(obs[batch_index], masks[batch_index])
                new_log_probs = dist.log_prob(actions[batch_index])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[batch_index])
                unclipped = ratio * advantages[batch_index]
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_range),
                    1.0 + float(self.config.clip_range),
                ) * advantages[batch_index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(new_values, value_targets[batch_index])
                loss = (
                    policy_loss
                    + float(self.config.value_coef) * value_loss
                    - float(self.config.entropy_coef) * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config.max_grad_norm))
                self.optimizer.step()

                clip_fraction = (
                    (torch.abs(ratio - 1.0) > float(self.config.clip_range))
                    .float()
                    .mean()
                )
                latest = {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "clip_fraction": float(clip_fraction.detach().cpu().item()),
                    "mean_reward": float(rewards.mean().detach().cpu().item()),
                }
                if self.policy.use_popart:
                    latest.update(self.policy.popart_metrics())
        return latest

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        next_indices: list[int | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        if next_indices is None:
            next_indices = [
                None if index + 1 >= int(rewards.shape[0]) else index + 1
                for index in range(int(rewards.shape[0]))
            ]
        for index in reversed(range(int(rewards.shape[0]))):
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + float(self.config.gamma) * next_values[index] * nonterminal - values[index]
            next_index = next_indices[index]
            next_advantage = (
                advantages[next_index]
                if next_index is not None and int(next_index) > int(index)
                else torch.tensor(0.0, dtype=torch.float32, device=self.device)
            )
            advantages[index] = (
                delta
                + float(self.config.gamma)
                * float(self.config.gae_lambda)
                * nonterminal
                * next_advantage
            )
        returns = advantages + values
        return advantages, returns

    @staticmethod
    def _build_next_indices(transitions: list[RolloutTransition]) -> list[int | None]:
        index_by_key = {
            (int(item.episode_index), int(item.vehicle_id)): index
            for index, item in enumerate(transitions)
        }
        next_indices: list[int | None] = []
        for item in transitions:
            if item.next_vehicle_id is None or bool(item.done):
                next_indices.append(None)
                continue
            next_indices.append(index_by_key.get((int(item.episode_index), int(item.next_vehicle_id))))
        return next_indices
