from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from chargingpilot.environment.structured_observation import StructuredObservation
from chargingpilot.routing.models import (
    HierarchicalAction,
    LambdaDecisionContext,
    S1DecisionContext,
    S2DecisionContext,
)


@dataclass(frozen=True)
class EncodedObservation:
    request: torch.Tensor
    stations: torch.Tensor
    pooled: torch.Tensor
    value: torch.Tensor
    is_single: bool

    @property
    def batch_size(self) -> int:
        return int(self.request.shape[0])


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: torch.Tensor
    value: torch.Tensor
    s1_entropy: torch.Tensor
    s2_entropy: torch.Tensor
    lambda_entropy: torch.Tensor


@dataclass(frozen=True)
class PolicySample(PolicyEvaluation):
    action: HierarchicalAction
    s2_context: S2DecisionContext
    lambda_context: LambdaDecisionContext | None


class HierarchicalActorCritic(nn.Module):
    """Three-head masked actor-critic for production structured observations."""

    REQUEST_FEATURES = 16
    STATION_COUNT = 72
    STATION_FEATURES = 33
    S2_FEATURES = 6
    LAMBDA_FEATURES = 5
    LAMBDA_COUNT = 15
    NODE_EMBEDDING_DIM = 32
    HIDDEN_DIM = 128

    def __init__(
        self,
        *,
        node_count: int,
        station_node_indices: Sequence[int],
    ) -> None:
        super().__init__()
        if type(node_count) is not int or node_count <= 0:
            raise ValueError("node_count must be a positive integer")
        station_indices = tuple(station_node_indices)
        if len(station_indices) != self.STATION_COUNT:
            raise ValueError(
                f"station_node_indices must contain {self.STATION_COUNT} entries"
            )
        if any(type(index) is not int for index in station_indices):
            raise ValueError("station_node_indices must contain integers")
        if any(not 0 <= index < node_count for index in station_indices):
            raise ValueError("station_node_indices contain an out-of-range node index")
        if any(
            left >= right
            for left, right in zip(station_indices, station_indices[1:])
        ):
            raise ValueError("station_node_indices must be strictly ascending")
        self.node_count = node_count
        self.s1_count = self.STATION_COUNT
        self.s2_count = self.STATION_COUNT + 1
        self.lambda_count = self.LAMBDA_COUNT

        self.node_embedding = nn.Embedding(node_count, self.NODE_EMBEDDING_DIM)
        self.register_buffer(
            "station_node_indices",
            torch.tensor(station_indices, dtype=torch.long),
        )
        self.request_encoder = nn.Sequential(
            nn.Linear(
                self.REQUEST_FEATURES + 2 * self.NODE_EMBEDDING_DIM,
                self.HIDDEN_DIM,
            ),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.Tanh(),
        )
        self.station_encoder = nn.Sequential(
            nn.Linear(
                self.STATION_FEATURES + self.NODE_EMBEDDING_DIM,
                self.HIDDEN_DIM,
            ),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.Tanh(),
        )
        self.station_attention = nn.MultiheadAttention(
            embed_dim=self.HIDDEN_DIM,
            num_heads=4,
            batch_first=True,
        )
        self.none_station_embedding = nn.Parameter(torch.empty(self.HIDDEN_DIM))
        nn.init.normal_(self.none_station_embedding, mean=0.0, std=0.02)

        self.s1_head = _score_head(3 * self.HIDDEN_DIM)
        self.s2_head = _score_head(4 * self.HIDDEN_DIM + self.S2_FEATURES)
        self.lambda_head = nn.Sequential(
            nn.Linear(4 * self.HIDDEN_DIM + self.LAMBDA_FEATURES, self.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, self.lambda_count),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(self.HIDDEN_DIM, 1),
        )

    def encode(
        self,
        observation: StructuredObservation | Sequence[StructuredObservation],
    ) -> EncodedObservation:
        observations, is_single = _observation_batch(observation)
        for item in observations:
            _validate_observation(item, self.node_count)

        device = self.node_embedding.weight.device
        dtype = self.node_embedding.weight.dtype
        request = torch.as_tensor(
            np.stack([item.request for item in observations]),
            dtype=dtype,
            device=device,
        )
        station_rows = torch.as_tensor(
            np.stack([item.stations for item in observations]),
            dtype=dtype,
            device=device,
        )
        origins = torch.tensor(
            [item.origin_index for item in observations],
            dtype=torch.long,
            device=device,
        )
        destinations = torch.tensor(
            [item.destination_index for item in observations],
            dtype=torch.long,
            device=device,
        )
        request_embedding = self.request_encoder(
            torch.cat(
                (
                    request,
                    self.node_embedding(origins),
                    self.node_embedding(destinations),
                ),
                dim=-1,
            )
        )
        station_nodes = self.node_embedding(self.station_node_indices).unsqueeze(0)
        station_nodes = station_nodes.expand(len(observations), -1, -1)
        station_embeddings = self.station_encoder(
            torch.cat((station_rows, station_nodes), dim=-1)
        )
        pooled, _attention_weights = self.station_attention(
            request_embedding.unsqueeze(1),
            station_embeddings,
            station_embeddings,
            need_weights=False,
        )
        pooled = pooled.squeeze(1)
        value = self.value_head(
            torch.cat((request_embedding, pooled), dim=-1)
        ).squeeze(-1)
        return EncodedObservation(
            request=request_embedding,
            stations=station_embeddings,
            pooled=pooled,
            value=value,
            is_single=is_single,
        )

    def s1_distribution(
        self,
        encoded: EncodedObservation,
        mask: np.ndarray | torch.Tensor,
    ) -> Categorical:
        batch = encoded.batch_size
        request = encoded.request.unsqueeze(1).expand(-1, self.s1_count, -1)
        pooled = encoded.pooled.unsqueeze(1).expand(-1, self.s1_count, -1)
        logits = self.s1_head(
            torch.cat((request, pooled, encoded.stations), dim=-1)
        ).squeeze(-1)
        return _masked_categorical(
            logits,
            _bool_batch(mask, batch, self.s1_count, logits.device),
            "s1 mask",
        )

    def s2_distribution(
        self,
        encoded: EncodedObservation,
        s1_indices: torch.Tensor | np.ndarray | Sequence[int],
        pair_features: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
    ) -> Categorical:
        batch = encoded.batch_size
        s1 = _index_batch(
            s1_indices, batch, self.s1_count, encoded.request.device, "s1 indices"
        )
        features = _float_batch(
            pair_features,
            batch,
            (self.s2_count, self.S2_FEATURES),
            encoded.request.device,
            encoded.request.dtype,
            "s2 features",
        )
        candidates = torch.cat(
            (
                encoded.stations,
                self.none_station_embedding.to(
                    device=encoded.request.device,
                    dtype=encoded.request.dtype,
                )
                .view(1, 1, -1)
                .expand(batch, 1, -1),
            ),
            dim=1,
        )
        selected_s1 = encoded.stations[
            torch.arange(batch, device=encoded.request.device), s1
        ]
        logits = self.s2_head(
            torch.cat(
                (
                    encoded.request.unsqueeze(1).expand(-1, self.s2_count, -1),
                    encoded.pooled.unsqueeze(1).expand(-1, self.s2_count, -1),
                    selected_s1.unsqueeze(1).expand(-1, self.s2_count, -1),
                    candidates,
                    features,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        return _masked_categorical(
            logits,
            _bool_batch(mask, batch, self.s2_count, logits.device),
            "s2 mask",
        )

    def lambda_distribution(
        self,
        encoded: EncodedObservation,
        s1_indices: torch.Tensor | np.ndarray | Sequence[int],
        s2_indices: torch.Tensor | np.ndarray | Sequence[int],
        lambda_features: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
    ) -> Categorical:
        batch = encoded.batch_size
        s1 = _index_batch(
            s1_indices, batch, self.s1_count, encoded.request.device, "s1 indices"
        )
        s2 = _index_batch(
            s2_indices, batch, self.s1_count, encoded.request.device, "s2 indices"
        )
        features = _float_batch(
            lambda_features,
            batch,
            (self.LAMBDA_FEATURES,),
            encoded.request.device,
            encoded.request.dtype,
            "lambda features",
        )
        rows = torch.arange(batch, device=encoded.request.device)
        logits = self.lambda_head(
            torch.cat(
                (
                    encoded.request,
                    encoded.pooled,
                    encoded.stations[rows, s1],
                    encoded.stations[rows, s2],
                    features,
                ),
                dim=-1,
            )
        )
        return _masked_categorical(
            logits,
            _bool_batch(mask, batch, self.lambda_count, logits.device),
            "lambda mask",
        )

    def sample_action(
        self,
        observation: StructuredObservation,
        s1_context: S1DecisionContext,
        context_provider: Any,
        *,
        deterministic: bool = False,
    ) -> PolicySample:
        _validate_s1_context(s1_context, self.s1_count)
        encoded = self.encode(observation)
        s1_dist = self.s1_distribution(encoded, s1_context.mask)
        s1_tensor = _select(s1_dist, deterministic)
        s1_index = int(s1_tensor.item())

        s2_context = context_provider.build_s2_context(
            s1_context.request_context, s1_index
        )
        _validate_s2_context(s1_context, s2_context, s1_index, self.s2_count)
        s2_dist = self.s2_distribution(
            encoded, s1_tensor, s2_context.features, s2_context.mask
        )
        s2_tensor = _select(s2_dist, deterministic)
        s2_index = int(s2_tensor.item())

        joint_log_prob = s1_dist.log_prob(s1_tensor) + s2_dist.log_prob(s2_tensor)
        lambda_context: LambdaDecisionContext | None = None
        lambda_entropy = encoded.value.new_zeros((1,))
        lambda_index: int | None = None
        if s2_index != self.s1_count:
            lambda_context = context_provider.build_lambda_context(
                s1_context.request_context, s1_index, s2_index
            )
            _validate_lambda_context(
                s1_context,
                s2_context,
                lambda_context,
                s1_index,
                s2_index,
                self.lambda_count,
            )
            lambda_dist = self.lambda_distribution(
                encoded,
                s1_tensor,
                s2_tensor,
                lambda_context.features,
                lambda_context.mask,
            )
            lambda_tensor = _select(lambda_dist, deterministic)
            lambda_index = int(lambda_tensor.item())
            joint_log_prob = joint_log_prob + lambda_dist.log_prob(lambda_tensor)
            lambda_entropy = lambda_dist.entropy()

        return PolicySample(
            action=HierarchicalAction(s1_index, s2_index, lambda_index),
            log_prob=joint_log_prob.squeeze(0),
            value=encoded.value.squeeze(0),
            s1_entropy=s1_dist.entropy().squeeze(0),
            s2_entropy=s2_dist.entropy().squeeze(0),
            lambda_entropy=lambda_entropy.squeeze(0),
            s2_context=s2_context,
            lambda_context=lambda_context,
        )

    def evaluate_action(
        self,
        observation: StructuredObservation | Sequence[StructuredObservation],
        action: HierarchicalAction | Sequence[HierarchicalAction],
        s1_context: S1DecisionContext | Sequence[S1DecisionContext],
        s2_context: S2DecisionContext | Sequence[S2DecisionContext],
        lambda_context: LambdaDecisionContext
        | None
        | Sequence[LambdaDecisionContext | None],
    ) -> PolicyEvaluation:
        actions, action_single = _item_batch(action, HierarchicalAction, "action")
        s1_contexts, s1_single = _item_batch(
            s1_context, S1DecisionContext, "s1_context"
        )
        s2_contexts, s2_single = _item_batch(
            s2_context, S2DecisionContext, "s2_context"
        )
        lambda_contexts = _optional_context_batch(lambda_context, action_single)
        if not (action_single == s1_single == s2_single):
            raise ValueError("action and decision-context batch shapes must match")
        batch = len(actions)
        if not (
            len(s1_contexts)
            == len(s2_contexts)
            == len(lambda_contexts)
            == batch
        ):
            raise ValueError("action and decision-context batch sizes must match")

        for index in range(batch):
            _validate_action_contexts(
                actions[index],
                s1_contexts[index],
                s2_contexts[index],
                lambda_contexts[index],
                self.s1_count,
                self.s2_count,
                self.lambda_count,
            )

        encoded = self.encode(observation)
        if encoded.batch_size != batch:
            raise ValueError("observation and action batch sizes must match")
        device = encoded.request.device
        s1_indices = torch.tensor(
            [item.s1_index for item in actions], dtype=torch.long, device=device
        )
        s2_indices = torch.tensor(
            [item.s2_index for item in actions], dtype=torch.long, device=device
        )
        s1_masks = np.stack([item.mask for item in s1_contexts])
        s2_masks = np.stack([item.mask for item in s2_contexts])
        s2_features = np.stack([item.features for item in s2_contexts])

        s1_dist = self.s1_distribution(encoded, s1_masks)
        s2_dist = self.s2_distribution(
            encoded, s1_indices, s2_features, s2_masks
        )
        log_prob = s1_dist.log_prob(s1_indices) + s2_dist.log_prob(s2_indices)
        lambda_entropy = torch.zeros_like(log_prob)

        split_rows = [
            index for index, item in enumerate(actions) if item.s2_index != self.s1_count
        ]
        if split_rows:
            row_tensor = torch.tensor(split_rows, dtype=torch.long, device=device)
            split_encoded = _select_encoded(encoded, row_tensor)
            split_contexts = [lambda_contexts[index] for index in split_rows]
            lambda_dist = self.lambda_distribution(
                split_encoded,
                s1_indices[row_tensor],
                s2_indices[row_tensor],
                np.stack([item.features for item in split_contexts]),
                np.stack([item.mask for item in split_contexts]),
            )
            lambda_indices = torch.tensor(
                [actions[index].lambda_index for index in split_rows],
                dtype=torch.long,
                device=device,
            )
            log_prob = log_prob.scatter_add(
                0, row_tensor, lambda_dist.log_prob(lambda_indices)
            )
            lambda_entropy = lambda_entropy.scatter(
                0, row_tensor, lambda_dist.entropy()
            )

        result = PolicyEvaluation(
            log_prob=log_prob,
            value=encoded.value,
            s1_entropy=s1_dist.entropy(),
            s2_entropy=s2_dist.entropy(),
            lambda_entropy=lambda_entropy,
        )
        if action_single:
            return PolicyEvaluation(
                log_prob=result.log_prob.squeeze(0),
                value=result.value.squeeze(0),
                s1_entropy=result.s1_entropy.squeeze(0),
                s2_entropy=result.s2_entropy.squeeze(0),
                lambda_entropy=result.lambda_entropy.squeeze(0),
            )
        return result


def _score_head(input_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_features, HierarchicalActorCritic.HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HierarchicalActorCritic.HIDDEN_DIM, 1),
    )


def _observation_batch(
    value: StructuredObservation | Sequence[StructuredObservation],
) -> tuple[tuple[StructuredObservation, ...], bool]:
    if isinstance(value, StructuredObservation):
        return (value,), True
    if not isinstance(value, Sequence) or not value:
        raise ValueError("observation must be a nonempty structured-observation batch")
    result = tuple(value)
    if not all(isinstance(item, StructuredObservation) for item in result):
        raise ValueError("observation batch contains an invalid item")
    return result, False


def _validate_observation(item: StructuredObservation, node_count: int) -> None:
    if not isinstance(item.request, np.ndarray) or item.request.dtype != np.float32:
        raise ValueError("observation request must be a float32 numpy array")
    if item.request.shape != (16,) or not np.isfinite(item.request).all():
        raise ValueError("observation request must have finite shape (16,)")
    if not isinstance(item.stations, np.ndarray) or item.stations.dtype != np.float32:
        raise ValueError("observation stations must be a float32 numpy array")
    if item.stations.shape != (72, 33) or not np.isfinite(item.stations).all():
        raise ValueError("observation stations must have finite shape (72, 33)")
    for name, index in (
        ("origin_index", item.origin_index),
        ("destination_index", item.destination_index),
    ):
        if type(index) is not int or not 0 <= index < node_count:
            raise ValueError(f"{name} is outside the node embedding range")


def _bool_batch(
    value: np.ndarray | torch.Tensor,
    batch: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        if value.dtype != np.bool_:
            raise ValueError("mask must use bool dtype")
        tensor = torch.from_numpy(value.copy())
    elif isinstance(value, torch.Tensor):
        if value.dtype != torch.bool:
            raise ValueError("mask must use bool dtype")
        tensor = value
    else:
        raise ValueError("mask must be a bool numpy array or tensor")
    if tuple(tensor.shape) == (width,):
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (batch, width):
        raise ValueError(f"mask must have shape ({batch}, {width})")
    return tensor.to(device=device)


def _float_batch(
    value: np.ndarray | torch.Tensor,
    batch: int,
    item_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        if value.dtype != np.float32:
            raise ValueError(f"{name} must use float32 dtype")
        tensor = torch.from_numpy(value.copy())
    elif isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise ValueError(f"{name} must use floating dtype")
        tensor = value
    else:
        raise ValueError(f"{name} must be a numpy array or tensor")
    if tuple(tensor.shape) == item_shape:
        tensor = tensor.unsqueeze(0)
    expected = (batch, *item_shape)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor.to(device=device, dtype=dtype)


def _index_batch(
    value: torch.Tensor | np.ndarray | Sequence[int],
    batch: int,
    upper_bound: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (batch,):
        raise ValueError(f"{name} must have shape ({batch},)")
    if bool(((tensor < 0) | (tensor >= upper_bound)).any()):
        raise ValueError(f"{name} contain an out-of-range index")
    return tensor


def _masked_categorical(
    logits: torch.Tensor, mask: torch.Tensor, name: str
) -> Categorical:
    if logits.shape != mask.shape:
        raise ValueError(f"{name} does not match logits shape")
    if not torch.isfinite(logits).all():
        raise ValueError("policy logits must be finite")
    if bool((~mask.any(dim=-1)).any()):
        raise ValueError(f"{name} must contain at least one feasible action per row")
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


def _select(distribution: Categorical, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return distribution.logits.argmax(dim=-1)
    return distribution.sample()


def _validate_s1_context(context: S1DecisionContext, station_count: int) -> None:
    if len(context.request_context.station_ids) != station_count:
        raise ValueError(f"policy requires exactly {station_count} stations")


def _validate_s2_context(
    parent: S1DecisionContext,
    child: S2DecisionContext,
    s1_index: int,
    s2_count: int,
) -> None:
    if child.request_context is not parent.request_context:
        raise ValueError("s2 context does not share the s1 request context")
    if child.s1_index != s1_index:
        raise ValueError("s2 context does not match selected s1 index")
    if child.mask.shape != (s2_count,):
        raise ValueError("s2 context has the wrong production shape")


def _validate_lambda_context(
    parent: S1DecisionContext,
    s2_context: S2DecisionContext,
    child: LambdaDecisionContext | None,
    s1_index: int,
    s2_index: int,
    lambda_count: int,
) -> None:
    if child is None:
        raise ValueError("split action requires a lambda context")
    if child.request_context is not parent.request_context:
        raise ValueError("lambda context does not share the s1 request context")
    if child.request_context is not s2_context.request_context:
        raise ValueError("lambda and s2 contexts do not share a request context")
    if (child.s1_index, child.s2_index) != (s1_index, s2_index):
        raise ValueError("lambda context does not match selected station indices")
    if child.mask.shape != (lambda_count,):
        raise ValueError("lambda context has the wrong production shape")


def _validate_action_contexts(
    action: HierarchicalAction,
    s1_context: S1DecisionContext,
    s2_context: S2DecisionContext,
    lambda_context: LambdaDecisionContext | None,
    s1_count: int,
    s2_count: int,
    lambda_count: int,
) -> None:
    _validate_s1_context(s1_context, s1_count)
    if type(action.s1_index) is not int or not 0 <= action.s1_index < s1_count:
        raise ValueError("action s1 index is out of range")
    if not bool(s1_context.mask[action.s1_index]):
        raise ValueError("action s1 index is masked")
    _validate_s2_context(s1_context, s2_context, action.s1_index, s2_count)
    if type(action.s2_index) is not int or not 0 <= action.s2_index < s2_count:
        raise ValueError("action s2 index is out of range")
    if not bool(s2_context.mask[action.s2_index]):
        raise ValueError("action s2 index is masked")
    if action.s2_index == s1_count:
        if action.lambda_index is not None or lambda_context is not None:
            raise ValueError("single-stop action must omit lambda")
        return
    if type(action.lambda_index) is not int or not 0 <= action.lambda_index < lambda_count:
        raise ValueError("split action lambda index is out of range")
    _validate_lambda_context(
        s1_context,
        s2_context,
        lambda_context,
        action.s1_index,
        action.s2_index,
        lambda_count,
    )
    if not bool(lambda_context.mask[action.lambda_index]):
        raise ValueError("action lambda index is masked")


def _item_batch(value: Any, expected_type: type, name: str) -> tuple[tuple[Any, ...], bool]:
    if isinstance(value, expected_type):
        return (value,), True
    if not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a nonempty batch")
    result = tuple(value)
    if not all(isinstance(item, expected_type) for item in result):
        raise ValueError(f"{name} batch contains an invalid item")
    return result, False


def _optional_context_batch(
    value: LambdaDecisionContext | None | Sequence[LambdaDecisionContext | None],
    action_single: bool,
) -> tuple[LambdaDecisionContext | None, ...]:
    if action_single:
        if value is not None and not isinstance(value, LambdaDecisionContext):
            raise ValueError("lambda_context has an invalid type")
        return (value,)
    if not isinstance(value, Sequence) or not value:
        raise ValueError("lambda_context must be a nonempty batch")
    result = tuple(value)
    if not all(item is None or isinstance(item, LambdaDecisionContext) for item in result):
        raise ValueError("lambda_context batch contains an invalid item")
    return result


def _select_encoded(
    encoded: EncodedObservation, rows: torch.Tensor
) -> EncodedObservation:
    return EncodedObservation(
        request=encoded.request[rows],
        stations=encoded.stations[rows],
        pooled=encoded.pooled[rows],
        value=encoded.value[rows],
        is_single=False,
    )
