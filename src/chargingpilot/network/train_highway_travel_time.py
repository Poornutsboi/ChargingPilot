from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from chargingpilot.network.GCT import (
    GraphConvolutionalTransformer,
    TravelTimeBatch,
    TravelTimeModelConfig,
)
from chargingpilot.network.highway_travel_time import (
    HighwayTravelTimeScenario,
    build_default_highway_scenario,
    write_training_csv,
)


DEFAULT_DATA_DIR = Path("datasets/traffic")
DEFAULT_DATASET_NAME = "highway_travel_time.csv"
DEFAULT_MODEL_OUTPUT = Path("models/highway_travel_time_gct.pt")
SEGMENT_FEATURE_DIM = 5
DEPARTURE_FEATURE_DIM = 5


@dataclass(frozen=True)
class HighwayTravelTimeTrainingResult:
    dataset_path: Path
    model_path: Path
    final_loss: float
    sample_count: int


def build_training_batch(
    scenario: HighwayTravelTimeScenario,
    rows: Sequence[Mapping[str, object]],
) -> tuple[TravelTimeBatch, torch.Tensor]:
    if not rows:
        raise ValueError("rows must contain at least one training sample.")

    segment_ids = [segment.segment_id for segment in scenario.segments]
    segment_index = {segment_id: index for index, segment_id in enumerate(segment_ids)}
    max_route_len = len(scenario.segments)

    route_segment_ids: list[list[int]] = []
    route_mask: list[list[bool]] = []
    departure_features: list[list[float]] = []
    targets: list[float] = []

    for row in rows:
        route_ids = _parse_route_segment_ids(row["route_segment_ids"])
        if not route_ids:
            raise ValueError("route_segment_ids must contain at least one segment id.")
        if len(route_ids) > max_route_len:
            raise ValueError("route length exceeds scenario max_route_len.")

        indices = [segment_index[segment_id] for segment_id in route_ids]
        padding = [0] * (max_route_len - len(indices))
        route_segment_ids.append(indices + padding)
        route_mask.append([True] * len(indices) + [False] * len(padding))
        departure_features.append(_departure_features(row))
        targets.append(float(row["actual_travel_time_min"]))

    edge_sources, edge_targets = scenario.edge_index()
    return (
        TravelTimeBatch(
            segment_features=_segment_features(scenario),
            edge_index=torch.tensor([edge_sources, edge_targets], dtype=torch.long),
            route_segment_ids=torch.tensor(route_segment_ids, dtype=torch.long),
            route_mask=torch.tensor(route_mask, dtype=torch.bool),
            departure_features=torch.tensor(departure_features, dtype=torch.float32),
        ),
        torch.tensor(targets, dtype=torch.float32),
    )


def train_highway_travel_time(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_output: str | Path = DEFAULT_MODEL_OUTPUT,
    samples: int = 2000,
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 42,
    learning_rate: float = 1e-3,
    embedding_dim: int = 32,
    num_heads: int = 4,
    transformer_layers: int = 2,
    feedforward_dim: int = 64,
    hidden_dim: int = 32,
    dropout: float = 0.05,
) -> HighwayTravelTimeTrainingResult:
    if int(samples) <= 0:
        raise ValueError("samples must be > 0.")
    if int(epochs) <= 0:
        raise ValueError("epochs must be > 0.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0.")

    torch.manual_seed(int(seed))
    scenario = build_default_highway_scenario()
    data_path = Path(data_dir) / DEFAULT_DATASET_NAME
    write_training_csv(
        scenario,
        data_path,
        sample_count=int(samples),
        seed=int(seed),
    )

    rows = _read_training_rows(data_path)
    full_batch, targets = build_training_batch(scenario, rows)
    config = TravelTimeModelConfig(
        segment_feature_dim=SEGMENT_FEATURE_DIM,
        max_route_len=len(scenario.segments),
        departure_feature_dim=DEPARTURE_FEATURE_DIM,
        embedding_dim=int(embedding_dim),
        num_heads=int(num_heads),
        transformer_layers=int(transformer_layers),
        feedforward_dim=int(feedforward_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
    model = GraphConvolutionalTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(seed))

    final_loss = math.inf
    for _epoch in range(int(epochs)):
        permutation = torch.randperm(int(targets.shape[0]), generator=generator)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, int(targets.shape[0]), int(batch_size)):
            batch_indices = permutation[start : start + int(batch_size)]
            batch = _slice_batch(full_batch, batch_indices)
            target = targets[batch_indices]

            optimizer.zero_grad()
            predictions = model(batch)
            loss = F.mse_loss(predictions, target)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(target.shape[0])
            epoch_loss += float(loss.item()) * batch_size_actual
            seen += batch_size_actual
        final_loss = epoch_loss / max(1, seen)

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "dataset_path": str(data_path),
            "station_chain": list(scenario.station_chain),
            "segment_ids": [segment.segment_id for segment in scenario.segments],
            "sample_count": int(samples),
            "final_loss": float(final_loss),
        },
        model_path,
    )
    return HighwayTravelTimeTrainingResult(
        dataset_path=data_path,
        model_path=model_path,
        final_loss=float(final_loss),
        sample_count=int(samples),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = train_highway_travel_time(
        data_dir=args.data_dir,
        model_output=args.output,
        samples=int(args.samples),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        embedding_dim=int(args.embedding_dim),
        num_heads=int(args.num_heads),
        transformer_layers=int(args.transformer_layers),
        feedforward_dim=int(args.feedforward_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    )
    print(
        "Trained highway travel-time model "
        f"on {result.sample_count} samples; "
        f"final_loss={result.final_loss:.6f}; "
        f"data={result.dataset_path}; "
        f"model={result.model_path}"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GCT highway travel-time model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser.parse_args(argv)


def _read_training_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _slice_batch(batch: TravelTimeBatch, indices: torch.Tensor) -> TravelTimeBatch:
    return TravelTimeBatch(
        segment_features=batch.segment_features,
        edge_index=batch.edge_index,
        route_segment_ids=batch.route_segment_ids[indices],
        route_mask=batch.route_mask[indices],
        departure_features=(
            None
            if batch.departure_features is None
            else batch.departure_features[indices]
        ),
    )


def _segment_features(scenario: HighwayTravelTimeScenario) -> torch.Tensor:
    rows: list[list[float]] = []
    for segment in scenario.segments:
        rows.append(
            [
                float(segment.length_km) / 50.0,
                float(segment.free_flow_speed_kmph) / 120.0,
                float(segment.lane_count) / 4.0,
                float(segment.bottleneck_multiplier),
                float(segment.free_flow_time_min) / 30.0,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def _departure_features(row: Mapping[str, object]) -> list[float]:
    departure_time = float(row["departure_time_min"])
    hour_angle = 2.0 * math.pi * ((departure_time % 1440.0) / 1440.0)
    return [
        math.sin(hour_angle),
        math.cos(hour_angle),
        float(row["peak_multiplier"]),
        float(row["weather_multiplier"]),
        float(row["incident_flag"]),
    ]


def _parse_route_segment_ids(value: object) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split(";") if item]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError("route_segment_ids must be a semicolon string or sequence.")


if __name__ == "__main__":
    main()
