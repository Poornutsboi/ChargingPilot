from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


@dataclass(frozen=True)
class TravelTimeModelConfig:
    segment_feature_dim: int
    max_route_len: int
    departure_feature_dim: int = 0
    embedding_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 3
    feedforward_dim: int = 128
    hidden_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if int(self.segment_feature_dim) <= 0:
            raise ValueError("segment_feature_dim must be > 0.")
        if int(self.max_route_len) <= 0:
            raise ValueError("max_route_len must be > 0.")
        if int(self.departure_feature_dim) < 0:
            raise ValueError("departure_feature_dim must be >= 0.")
        if int(self.embedding_dim) <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be > 0.")
        if int(self.embedding_dim) % int(self.num_heads) != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if int(self.transformer_layers) <= 0:
            raise ValueError("transformer_layers must be > 0.")
        if int(self.feedforward_dim) <= 0:
            raise ValueError("feedforward_dim must be > 0.")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be >= 0 and < 1.")


@dataclass(frozen=True)
class TravelTimeBatch:
    segment_features: torch.Tensor
    edge_index: torch.Tensor
    route_segment_ids: torch.Tensor
    route_mask: torch.Tensor
    departure_features: torch.Tensor | None = None


class _TravelTimeModule(Protocol):
    training: bool

    def eval(self) -> _TravelTimeModule:
        ...

    def train(self, mode: bool = True) -> _TravelTimeModule:
        ...

    def __call__(self, batch: TravelTimeBatch) -> torch.Tensor:
        ...


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if int(in_features) <= 0:
            raise ValueError("in_features must be > 0.")
        if int(out_features) <= 0:
            raise ValueError("out_features must be > 0.")
        self.linear = nn.Linear(int(in_features), int(out_features), bias=bool(bias))

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(features, edge_index)

        edge_index = edge_index.to(device=features.device, dtype=torch.long)
        aggregated = features.clone()
        degree = torch.ones(
            features.shape[0],
            dtype=features.dtype,
            device=features.device,
        )

        if edge_index.numel() > 0:
            source = edge_index[0]
            target = edge_index[1]
            aggregated.index_add_(0, target, features[source])
            aggregated.index_add_(0, source, features[target])

            ones = torch.ones(source.shape[0], dtype=features.dtype, device=features.device)
            degree.index_add_(0, target, ones)
            degree.index_add_(0, source, ones)

        averaged = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        return self.linear(averaged)

    def _validate_inputs(self, features: torch.Tensor, edge_index: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape [num_segments, feature_dim].")
        if features.shape[0] <= 0:
            raise ValueError("features must contain at least one segment.")
        if not torch.is_floating_point(features):
            raise TypeError("features must be a floating point tensor.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_index.numel() == 0:
            return
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= int(features.shape[0]):
            raise ValueError("edge_index contains a segment id outside segment_features.")


class GraphConvolutionalTransformer(nn.Module):
    def __init__(self, config: TravelTimeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_embedding = nn.Linear(
            int(config.segment_feature_dim),
            int(config.embedding_dim),
        )
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvolutionLayer(int(config.embedding_dim), int(config.embedding_dim)),
                GraphConvolutionLayer(int(config.embedding_dim), int(config.embedding_dim)),
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(config.embedding_dim),
            nhead=int(config.num_heads),
            dim_feedforward=int(config.feedforward_dim),
            dropout=float(config.dropout),
            activation="relu",
            batch_first=True,
        )
        self.route_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.transformer_layers),
            enable_nested_tensor=False,
        )
        route_feature_dim = (
            int(config.max_route_len) * int(config.embedding_dim)
            + int(config.departure_feature_dim)
        )
        self.output_layers = nn.Sequential(
            nn.Linear(route_feature_dim, int(config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(config.hidden_dim), int(config.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(config.hidden_dim), 1),
        )

    def forward(self, batch: TravelTimeBatch) -> torch.Tensor:
        self._validate_batch(batch)

        segment_embeddings = self.input_embedding(batch.segment_features)
        for layer in self.gcn_layers:
            segment_embeddings = torch.relu(layer(segment_embeddings, batch.edge_index))

        route_segment_ids = batch.route_segment_ids.to(
            device=segment_embeddings.device,
            dtype=torch.long,
        )
        route_mask = batch.route_mask.to(device=segment_embeddings.device, dtype=torch.bool)
        route_embeddings = segment_embeddings[route_segment_ids]
        route_embeddings = route_embeddings.masked_fill(~route_mask.unsqueeze(-1), 0.0)

        encoded_route = self.route_encoder(
            route_embeddings,
            src_key_padding_mask=~route_mask,
        )
        encoded_route = encoded_route.masked_fill(~route_mask.unsqueeze(-1), 0.0)
        route_representation = encoded_route.reshape(encoded_route.shape[0], -1)

        departure_features = self._departure_features(batch, route_representation)
        if departure_features is not None:
            route_representation = torch.cat(
                [route_representation, departure_features],
                dim=1,
            )

        return self.output_layers(route_representation).squeeze(-1)

    def _departure_features(
        self,
        batch: TravelTimeBatch,
        route_representation: torch.Tensor,
    ) -> torch.Tensor | None:
        feature_dim = int(self.config.departure_feature_dim)
        if feature_dim == 0:
            return None
        if batch.departure_features is None:
            return torch.zeros(
                route_representation.shape[0],
                feature_dim,
                dtype=route_representation.dtype,
                device=route_representation.device,
            )
        return batch.departure_features.to(
            device=route_representation.device,
            dtype=route_representation.dtype,
        )

    def _validate_batch(self, batch: TravelTimeBatch) -> None:
        if batch.segment_features.ndim != 2:
            raise ValueError("segment_features must have shape [num_segments, segment_feature_dim].")
        if batch.segment_features.shape[1] != int(self.config.segment_feature_dim):
            raise ValueError("segment_features width must match segment_feature_dim.")
        if not torch.is_floating_point(batch.segment_features):
            raise TypeError("segment_features must be a floating point tensor.")

        if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if batch.edge_index.numel() > 0:
            min_index = int(batch.edge_index.min().item())
            max_index = int(batch.edge_index.max().item())
            if min_index < 0 or max_index >= int(batch.segment_features.shape[0]):
                raise ValueError("edge_index contains a segment id outside segment_features.")

        expected_route_shape = (None, int(self.config.max_route_len))
        if batch.route_segment_ids.ndim != 2:
            raise ValueError("route_segment_ids must have shape [batch, max_route_len].")
        if batch.route_segment_ids.shape[1] != expected_route_shape[1]:
            raise ValueError("route_segment_ids length must match max_route_len.")
        if batch.route_mask.shape != batch.route_segment_ids.shape:
            raise ValueError("route_mask must have the same shape as route_segment_ids.")
        if batch.route_segment_ids.shape[0] <= 0:
            raise ValueError("route batch must contain at least one route.")
        if not batch.route_mask.any(dim=1).all():
            raise ValueError("each route must contain at least one unmasked segment.")
        if batch.route_segment_ids.numel() > 0:
            min_route_id = int(batch.route_segment_ids.min().item())
            max_route_id = int(batch.route_segment_ids.max().item())
            if min_route_id < 0 or max_route_id >= int(batch.segment_features.shape[0]):
                raise ValueError("route_segment_ids contains a segment id outside segment_features.")

        departure_dim = int(self.config.departure_feature_dim)
        if batch.departure_features is None:
            return
        if batch.departure_features.ndim != 2:
            raise ValueError("departure_features must have shape [batch, departure_feature_dim].")
        if batch.departure_features.shape[0] != batch.route_segment_ids.shape[0]:
            raise ValueError("departure_features batch size must match route_segment_ids.")
        if batch.departure_features.shape[1] != departure_dim:
            raise ValueError("departure_features width must match departure_feature_dim.")
        if not torch.is_floating_point(batch.departure_features):
            raise TypeError("departure_features must be a floating point tensor.")


def predict_travel_time(
    model: _TravelTimeModule,
    batch: TravelTimeBatch,
) -> torch.Tensor:
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            return torch.clamp(model(batch), min=0.0)
    finally:
        model.train(was_training)
