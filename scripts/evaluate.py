from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from chargingpilot.simulator.models import ChargingSocRequest, StationSpec, VehicleSpec
from chargingpilot.simulator.simulator import SimulatorCore

from chargingpilot.evaluation import (
    BaselineSelection,
    DayBootstrapInterval,
    EVALUATION_CSV_FIELDS,
    EVALUATION_POLICY_LABELS,
    REQUEST_CSV_FIELDS,
    EvaluationEpisodeSpec,
    HierarchicalEvaluationAggregate,
    HierarchicalEvaluationMetadata,
    HierarchicalPolicyEvaluationReport,
    HierarchicalEvaluationRecord,
    StationVisit,
    aggregate_hierarchical_evaluation,
    bootstrap_day_confidence_interval,
    select_mandatory_service_shortest,
    select_minimum_wait_single,
    select_random_feasible,
    run_hierarchical_evaluation,
    write_hierarchical_evaluation_csv,
    write_hierarchical_evaluation_json,
    write_hierarchical_request_csv,
    write_hierarchical_request_json,
)


CAPACITY = [10, 25, 18, 12, 12, 22, 18]
DEFAULT_TRAIN_DIR = Path("datasets/train")
DEFAULT_OUTPUT_CSV = Path("evaluation_results/no_split_origin_station_metrics.csv")


@dataclass(frozen=True)
class VehicleRequestRow:
    vehicle_id: int
    arrival_time_min: float
    start_soc: float
    target_soc: float
    battery_capacity_kwh: float
    origin_station_id: int
    destination_station_id: int
    rho_kwh_per_km: float


@dataclass(frozen=True)
class StationMetric:
    episode: str
    station_id: int
    served: int
    queue_time_min: float
    mean_queue_length: float
    max_queue_length: int


def select_episode_paths(data_dir: str | Path, limit: int | None = 50) -> list[Path]:
    paths = sorted(Path(data_dir).glob("episode_*.csv"))
    if not paths:
        paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No episode CSV files found under {data_dir}.")
    if limit is None:
        return paths
    return paths[: int(limit)]


def evaluate_episode(
    episode_path: str | Path,
    *,
    capacities: Iterable[int] = CAPACITY,
    plug_kw: float = 120.0,
    eta: float = 0.95,
    timestep_minutes: float = 1.0,
    max_drain_minutes: int = 20000,
) -> list[StationMetric]:
    capacity_values = [int(value) for value in capacities]
    simulator = SimulatorCore(
        station_specs=_build_station_specs(
            capacities=capacity_values,
            plug_kw=float(plug_kw),
            eta=float(eta),
        ),
        timestep_minutes=float(timestep_minutes),
    )
    requests = _load_episode_requests(Path(episode_path), plug_kw=float(plug_kw))
    queue_samples: dict[int, list[int]] = {
        station_id: []
        for station_id in simulator.station_ids
    }

    request_index = 0
    while request_index < len(requests):
        arrival_time = float(requests[request_index].arrival_time_min)
        _advance_and_sample(
            simulator=simulator,
            target_time=arrival_time,
            queue_samples=queue_samples,
        )
        while (
            request_index < len(requests)
            and float(requests[request_index].arrival_time_min) == arrival_time
        ):
            simulator.enqueue_soc_arrival(
                _to_charging_request(
                    requests[request_index],
                    plug_kw=float(plug_kw),
                )
            )
            request_index += 1
        _sample_queue_lengths(simulator, queue_samples)

    drain_steps = 0
    while _has_station_work(simulator):
        if drain_steps >= int(max_drain_minutes):
            raise RuntimeError(
                f"Exceeded max_drain_minutes={max_drain_minutes} while draining {episode_path}."
            )
        simulator.advance_to(float(simulator.clock) + float(timestep_minutes))
        _sample_queue_lengths(simulator, queue_samples)
        drain_steps += 1

    metrics = simulator.get_metrics(query_time=simulator.clock)
    episode_name = Path(episode_path).stem
    return [
        StationMetric(
            episode=episode_name,
            station_id=int(station_id),
            served=int(metrics.ev_served[int(station_id)]),
            queue_time_min=float(metrics.queue_time[int(station_id)]),
            mean_queue_length=(
                float(mean(queue_samples[int(station_id)]))
                if queue_samples[int(station_id)]
                else 0.0
            ),
            max_queue_length=(
                int(max(queue_samples[int(station_id)]))
                if queue_samples[int(station_id)]
                else 0
            ),
        )
        for station_id in simulator.station_ids
    ]


def evaluate_dataset(
    data_dir: str | Path = DEFAULT_TRAIN_DIR,
    *,
    limit: int | None = 50,
    capacities: Iterable[int] = CAPACITY,
    plug_kw: float = 120.0,
    eta: float = 0.95,
    timestep_minutes: float = 1.0,
) -> list[StationMetric]:
    results: list[StationMetric] = []
    for episode_path in select_episode_paths(data_dir, limit=limit):
        results.extend(
            evaluate_episode(
                episode_path,
                capacities=capacities,
                plug_kw=float(plug_kw),
                eta=float(eta),
                timestep_minutes=float(timestep_minutes),
            )
        )
    return results


def write_station_metrics(metrics: list[StationMetric], output_csv: str | Path) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "station_id",
                "served",
                "queue_time_min",
                "mean_queue_length",
                "max_queue_length",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "episode": metric.episode,
                    "station_id": metric.station_id,
                    "served": metric.served,
                    "queue_time_min": f"{metric.queue_time_min:.6f}",
                    "mean_queue_length": f"{metric.mean_queue_length:.6f}",
                    "max_queue_length": metric.max_queue_length,
                }
            )
    return path


def summarize_station_metrics(metrics: list[StationMetric]) -> list[StationMetric]:
    by_station: dict[int, list[StationMetric]] = {}
    for metric in metrics:
        by_station.setdefault(int(metric.station_id), []).append(metric)

    summary: list[StationMetric] = []
    for station_id in sorted(by_station):
        items = by_station[station_id]
        summary.append(
            StationMetric(
                episode="mean_over_episodes",
                station_id=int(station_id),
                served=int(round(mean(item.served for item in items))),
                queue_time_min=float(mean(item.queue_time_min for item in items)),
                mean_queue_length=float(mean(item.mean_queue_length for item in items)),
                max_queue_length=int(max(item.max_queue_length for item in items)),
            )
        )
    return summary


def _build_station_specs(
    *,
    capacities: list[int],
    plug_kw: float,
    eta: float,
) -> list[StationSpec]:
    return [
        StationSpec(
            station_id=index + 1,
            charge_capacity=int(capacity),
            p_plug_kw=float(plug_kw),
            p_max_kw=float(capacity) * float(plug_kw),
            eta=float(eta),
        )
        for index, capacity in enumerate(capacities)
    ]


def _load_episode_requests(episode_path: Path, *, plug_kw: float) -> list[VehicleRequestRow]:
    rows: list[VehicleRequestRow] = []
    with episode_path.open(newline="", encoding="utf-8") as csv_file:
        for row_number, row in enumerate(csv.DictReader(csv_file), start=1):
            rows.append(
                VehicleRequestRow(
                    vehicle_id=_parse_vehicle_id(row["vehicle_id"], fallback=row_number),
                    arrival_time_min=_parse_hhmm_minutes(row["arrival_time"]),
                    start_soc=float(row["start_soc"]),
                    target_soc=float(row["target_soc"]),
                    battery_capacity_kwh=float(row["B_i"]),
                    origin_station_id=_parse_station_id(row["o_i"]),
                    destination_station_id=_parse_station_id(row["d_i"]),
                    rho_kwh_per_km=float(row["rho_i"]),
                )
            )
    return sorted(rows, key=lambda item: (float(item.arrival_time_min), int(item.vehicle_id)))


def _to_charging_request(row: VehicleRequestRow, *, plug_kw: float = 120.0) -> ChargingSocRequest:
    spec = VehicleSpec(
        battery_capacity=float(row.battery_capacity_kwh),
        initial_soc=float(row.start_soc),
        soc_min=0.0,
        p_max_kw=float(plug_kw),
        p_min_kw=30.0,
        rho_kwh_per_km=float(row.rho_kwh_per_km),
        origin=int(row.origin_station_id),
        destination=int(row.destination_station_id),
        departure_time=float(row.arrival_time_min),
        path_nodes=(int(row.origin_station_id), int(row.destination_station_id)),
        path_edges=(),
        candidate_stations=(int(row.origin_station_id),),
        demand_kwh=float(row.battery_capacity_kwh)
        * (float(row.target_soc) - float(row.start_soc)),
    )
    return ChargingSocRequest(
        vehicle_id=int(row.vehicle_id),
        station_id=int(row.origin_station_id),
        arrival_time=float(row.arrival_time_min),
        vehicle_spec=spec,
        arrival_soc=float(row.start_soc),
        target_soc=float(row.target_soc),
    )


def _advance_and_sample(
    *,
    simulator: SimulatorCore,
    target_time: float,
    queue_samples: dict[int, list[int]],
) -> None:
    while float(simulator.clock) < float(target_time):
        next_time = min(
            float(simulator.clock) + float(simulator.timestep_minutes),
            float(target_time),
        )
        simulator.advance_to(next_time)
        _sample_queue_lengths(simulator, queue_samples)


def _sample_queue_lengths(
    simulator: SimulatorCore,
    queue_samples: dict[int, list[int]],
) -> None:
    state = simulator.get_state(query_time=simulator.clock)
    for station_id in simulator.station_ids:
        station_state = state["stations"][int(station_id)]
        queue_samples[int(station_id)].append(
            int(len(station_state["queue_waiting_time"]))
        )


def _has_station_work(simulator: SimulatorCore) -> bool:
    for station in simulator._stations.values():  # noqa: SLF001 - evaluation helper
        snapshot = station.snapshot()
        if snapshot.waiting_queue or snapshot.active_sessions:
            return True
    return False


def _parse_hhmm_minutes(value: str) -> float:
    hour, minute = str(value).split(":", maxsplit=1)
    return float(int(hour) * 60 + int(minute))


def _parse_station_id(value: str) -> int:
    text = str(value).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return int(text)


def _parse_vehicle_id(value: str, *, fallback: int) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        return int(fallback)
    return int(digits)


def _parse_capacities(value: str) -> list[int]:
    capacities = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not capacities:
        raise ValueError("capacities must contain at least one value.")
    return capacities


def _print_summary(metrics: list[StationMetric], output_csv: Path) -> None:
    episodes = sorted({metric.episode for metric in metrics})
    print("No-split origin station evaluation")
    print(f"episodes: {len(episodes)}")
    print(f"rows: {len(metrics)}")
    print(f"output_csv: {output_csv}")
    print()
    print("station_id,mean_served,mean_queue_time_min,mean_queue_length,max_queue_length")
    for metric in summarize_station_metrics(metrics):
        print(
            f"{metric.station_id},"
            f"{metric.served},"
            f"{metric.queue_time_min:.6f},"
            f"{metric.mean_queue_length:.6f},"
            f"{metric.max_queue_length}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate no-split baseline: every vehicle charges at o_i."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--capacities",
        default=",".join(str(value) for value in CAPACITY),
        help="Comma-separated capacities for S1..Sn.",
    )
    parser.add_argument("--plug-kw", type=float, default=120.0)
    parser.add_argument("--eta", type=float, default=0.95)
    parser.add_argument("--timestep-minutes", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_dataset(
        args.data_dir,
        limit=int(args.limit),
        capacities=_parse_capacities(args.capacities),
        plug_kw=float(args.plug_kw),
        eta=float(args.eta),
        timestep_minutes=float(args.timestep_minutes),
    )
    output_csv = write_station_metrics(metrics, args.output_csv)
    _print_summary(metrics, output_csv)


if __name__ == "__main__":
    main()
