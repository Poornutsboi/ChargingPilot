from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DEFAULT_NODES_PATH = BASE_DIR / "nodes_final.geojson"
DEFAULT_LINKS_PATH = BASE_DIR / "links_final.geojson"
DEFAULT_OUTPUT_PATH = BASE_DIR / "charging_requests.csv"
SHORT_PATH_MODULE_PATH = BASE_DIR / "shortest_path.py"

TRAFFIC_FLOW = [
    275,
    196,
    172,
    239,
    546,
    1486,
    2891,
    2887,
    2179,
    2091,
    2298,
    2601,
    2561,
    2832,
    3206,
    3055,
    3313,
    2633,
    2258,
    1709,
    1394,
    1223,
    693,
    339,
]
BATTERY_VALUES = [45, 55, 65, 75, 90, 100, 110]
BATTERY_PROBS = [0.10, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03]
RHO_MEAN = 0.20
RHO_STD = 0.04
RHO_LOW = 0.12
RHO_HIGH = 0.32
TARGET_SOC_LOW = 0.75
TARGET_SOC_HIGH = 1.0
TARGET_SOC_BETA_A = 2.5
TARGET_SOC_BETA_B = 1.2
OUTPUT_COLUMNS = [
    "vehicle_id",
    "date",
    "datetime",
    "arrival_time",
    "start_soc",
    "target_soc",
    "B_i",
    "o_i",
    "d_i",
    "rho_i",
    "path_node_ids",
    "path_length_km",
    "nearest_service_id",
    "distance_to_nearest_service_km",
    "energy_to_nearest_service_kwh",
    "min_start_soc_required",
    "long_trip",
]


def generate_requests(
    *,
    nodes_path: Path = DEFAULT_NODES_PATH,
    links_path: Path = DEFAULT_LINKS_PATH,
    days: int = 1,
    requests_per_day: int = 800,
    seed: int | None = 42,
    min_long_trip_km: float = 120.0,
    long_trip_share: float = 0.82,
    safety_soc: float = 0.04,
    max_attempts_per_request: int = 300,
) -> list[dict[str, Any]]:
    if int(days) <= 0:
        raise ValueError("days must be positive")
    if int(requests_per_day) <= 0:
        raise ValueError("requests_per_day must be positive")
    if not 0.0 <= float(long_trip_share) <= 1.0:
        raise ValueError("long_trip_share must be in [0, 1]")

    rng = random.Random(seed)
    sp = _load_shortest_path_module()
    road_map = sp.load_road_map(nodes_path, links_path, undirected=True)
    service_ids = {node_id for node_id, node_type in road_map.node_types.items() if node_type == "service"}
    toll_ids = sorted(road_map.toll_ids)
    if not service_ids:
        raise ValueError("road_map has no service nodes")
    if len(toll_ids) < 2:
        raise ValueError("road_map needs at least two toll nodes")

    edge_lengths = _edge_lengths(road_map.graph)
    hour_weights = _normalize(TRAFFIC_FLOW)
    path_cache: dict[tuple[int, int], Any] = {}
    rows: list[dict[str, Any]] = []
    vehicle_counter = 1

    for day_index in range(int(days)):
        for _ in range(int(requests_per_day)):
            row = _sample_feasible_request(
                rng=rng,
                sp=sp,
                road_map=road_map,
                toll_ids=toll_ids,
                service_ids=service_ids,
                edge_lengths=edge_lengths,
                path_cache=path_cache,
                vehicle_index=vehicle_counter,
                day_index=day_index,
                hour_weights=hour_weights,
                min_long_trip_km=float(min_long_trip_km),
                long_trip_share=float(long_trip_share),
                safety_soc=float(safety_soc),
                max_attempts=int(max_attempts_per_request),
            )
            rows.append(row)
            vehicle_counter += 1

    return sorted(rows, key=lambda item: (item["date"], item["arrival_time"], item["vehicle_id"]))


def first_service_on_path(
    path_nodes: Sequence[int],
    service_ids: set[int],
    edge_lengths: dict[tuple[int, int], float],
) -> tuple[int, float]:
    distance_m = 0.0
    for left, right in zip(path_nodes, path_nodes[1:]):
        distance_m += _lookup_edge_length(edge_lengths, int(left), int(right))
        if int(right) in service_ids:
            return int(right), distance_m
    raise ValueError("path does not reach a service node")


def feasible_start_soc(
    *,
    battery_kwh: float,
    rho_kwh_per_km: float,
    distance_to_service_km: float,
    safety_soc: float,
    rng: random.Random,
) -> float:
    required = float(distance_to_service_km) * float(rho_kwh_per_km) / float(battery_kwh) + float(safety_soc)
    if required > 0.92:
        raise ValueError("vehicle cannot reach nearest service with this battery and consumption")
    upper = min(0.92, max(required + 0.02, required * 1.25))
    return round(float(rng.uniform(required, upper)), 3)


def build_request_row(
    *,
    vehicle_index: int,
    day_index: int,
    minute_of_day: int,
    origin: int,
    destination: int,
    path_nodes: Sequence[int],
    path_length_km: float,
    nearest_service_id: int,
    distance_to_service_km: float,
    battery_kwh: float,
    rho_kwh_per_km: float,
    start_soc: float,
    target_soc: float,
) -> dict[str, Any]:
    date = _date_for_day(int(day_index))
    arrival_time = _format_hhmm(int(minute_of_day))
    energy_to_service = float(distance_to_service_km) * float(rho_kwh_per_km)
    min_required = energy_to_service / float(battery_kwh)
    return {
        "vehicle_id": f"EV{int(vehicle_index):08d}",
        "date": date,
        "datetime": f"{date} {arrival_time}",
        "arrival_time": arrival_time,
        "start_soc": round(float(start_soc), 3),
        "target_soc": round(float(target_soc), 3),
        "B_i": int(battery_kwh),
        "o_i": int(origin),
        "d_i": int(destination),
        "rho_i": round(float(rho_kwh_per_km), 3),
        "path_node_ids": "-".join(str(int(node_id)) for node_id in path_nodes),
        "path_length_km": round(float(path_length_km), 3),
        "nearest_service_id": int(nearest_service_id),
        "distance_to_nearest_service_km": round(float(distance_to_service_km), 3),
        "energy_to_nearest_service_kwh": round(float(energy_to_service), 3),
        "min_start_soc_required": round(float(min_required), 3),
        "long_trip": int(float(path_length_km) >= 120.0),
    }


def write_requests(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    rows = generate_requests(
        nodes_path=args.nodes,
        links_path=args.links,
        days=int(args.days),
        requests_per_day=int(args.requests_per_day),
        seed=args.seed,
        min_long_trip_km=float(args.min_long_trip_km),
        long_trip_share=float(args.long_trip_share),
        safety_soc=float(args.safety_soc),
        max_attempts_per_request=int(args.max_attempts_per_request),
    )
    output_path = write_requests(rows, args.output)
    long_count = sum(int(row["long_trip"]) for row in rows)
    print(f"wrote: {output_path}")
    print(f"requests: {len(rows)}")
    print(f"date_range: {rows[0]['date']} to {rows[-1]['date']}")
    print(f"long_trip_count: {long_count} ({long_count / len(rows):.1%})")
    print(f"path_length_km_mean: {sum(float(row['path_length_km']) for row in rows) / len(rows):.2f}")
    print(
        "max_energy_to_service_kwh: "
        f"{max(float(row['energy_to_nearest_service_kwh']) for row in rows):.2f}"
    )


def _sample_feasible_request(
    *,
    rng: random.Random,
    sp: Any,
    road_map: Any,
    toll_ids: list[int],
    service_ids: set[int],
    edge_lengths: dict[tuple[int, int], float],
    path_cache: dict[tuple[int, int], Any],
    vehicle_index: int,
    day_index: int,
    hour_weights: list[float],
    min_long_trip_km: float,
    long_trip_share: float,
    safety_soc: float,
    max_attempts: int,
) -> dict[str, Any]:
    for _attempt in range(max_attempts):
        origin = rng.choice(toll_ids)
        destination = _sample_destination(
            rng=rng,
            origin=origin,
            toll_ids=toll_ids,
            road_map=road_map,
            sp=sp,
            path_cache=path_cache,
            prefer_long=rng.random() < long_trip_share,
            min_long_trip_km=min_long_trip_km,
        )
        if destination is None:
            continue
        result = _cached_path(sp, road_map, path_cache, origin, destination)
        path_nodes = result.node_ids
        path_length_km = float(result.distance_m) / 1000.0
        if path_length_km < 50.0:
            continue
        try:
            nearest_service_id, distance_to_service_m = first_service_on_path(path_nodes, service_ids, edge_lengths)
        except ValueError:
            continue

        battery_kwh = _sample_discrete(rng, BATTERY_VALUES, BATTERY_PROBS)
        rho = _sample_rho(rng)
        try:
            start_soc = feasible_start_soc(
                battery_kwh=float(battery_kwh),
                rho_kwh_per_km=float(rho),
                distance_to_service_km=distance_to_service_m / 1000.0,
                safety_soc=float(safety_soc),
                rng=rng,
            )
        except ValueError:
            continue

        target_soc = _sample_target_soc(rng)
        hour = _sample_discrete(rng, list(range(24)), hour_weights)
        minute_of_day = int(hour) * 60 + rng.randrange(0, 60)
        return build_request_row(
            vehicle_index=int(vehicle_index),
            day_index=int(day_index),
            minute_of_day=minute_of_day,
            origin=int(origin),
            destination=int(destination),
            path_nodes=path_nodes,
            path_length_km=path_length_km,
            nearest_service_id=int(nearest_service_id),
            distance_to_service_km=distance_to_service_m / 1000.0,
            battery_kwh=float(battery_kwh),
            rho_kwh_per_km=float(rho),
            start_soc=float(start_soc),
            target_soc=float(target_soc),
        )
    raise RuntimeError(f"unable to sample a feasible request after {max_attempts} attempts")


def _sample_destination(
    *,
    rng: random.Random,
    origin: int,
    toll_ids: list[int],
    road_map: Any,
    sp: Any,
    path_cache: dict[tuple[int, int], Any],
    prefer_long: bool,
    min_long_trip_km: float,
) -> int | None:
    candidates = rng.sample([node_id for node_id in toll_ids if node_id != origin], k=min(80, len(toll_ids) - 1))
    viable: list[tuple[int, float]] = []
    for destination in candidates:
        try:
            result = _cached_path(sp, road_map, path_cache, origin, destination)
        except ValueError:
            continue
        distance_km = float(result.distance_m) / 1000.0
        if prefer_long and distance_km < float(min_long_trip_km):
            continue
        viable.append((destination, distance_km))
    if not viable:
        return None
    if prefer_long:
        viable.sort(key=lambda item: item[1], reverse=True)
        pool = viable[: max(1, len(viable) // 3)]
        return rng.choice(pool)[0]
    return rng.choice(viable)[0]


def _cached_path(sp: Any, road_map: Any, path_cache: dict[tuple[int, int], Any], origin: int, destination: int) -> Any:
    key = (int(origin), int(destination))
    if key not in path_cache:
        path_cache[key] = sp.shortest_path_between_tolls(road_map, int(origin), int(destination))
    return path_cache[key]


def _edge_lengths(graph: dict[int, list[tuple[int, float]]]) -> dict[tuple[int, int], float]:
    lengths: dict[tuple[int, int], float] = {}
    for source, edges in graph.items():
        for target, length in edges:
            key = (int(source), int(target))
            reverse_key = (int(target), int(source))
            value = float(length)
            lengths[key] = min(value, lengths.get(key, value))
            lengths[reverse_key] = min(value, lengths.get(reverse_key, value))
    return lengths


def _lookup_edge_length(edge_lengths: dict[tuple[int, int], float], left: int, right: int) -> float:
    try:
        return float(edge_lengths[(int(left), int(right))])
    except KeyError as exc:
        raise ValueError(f"missing edge length for {left}->{right}") from exc


def _sample_rho(rng: random.Random) -> float:
    while True:
        value = rng.gauss(RHO_MEAN, RHO_STD)
        if RHO_LOW <= value <= RHO_HIGH:
            return float(value)


def _sample_target_soc(rng: random.Random) -> float:
    x = rng.betavariate(TARGET_SOC_BETA_A, TARGET_SOC_BETA_B)
    return TARGET_SOC_LOW + (TARGET_SOC_HIGH - TARGET_SOC_LOW) * x


def _sample_discrete(rng: random.Random, values: Sequence[Any], weights: Sequence[float]) -> Any:
    return rng.choices(list(values), weights=list(weights), k=1)[0]


def _normalize(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [float(value) / total for value in values]


def _date_for_day(day_index: int) -> str:
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_number = int(day_index)
    month = 1
    for length in month_lengths:
        if day_number < length:
            return f"{month:02d}-{day_number + 1:02d}"
        day_number -= length
        month += 1
    return f"12-31"


def _format_hhmm(minute_of_day: int) -> str:
    minute = int(minute_of_day) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _load_shortest_path_module() -> Any:
    spec = importlib.util.spec_from_file_location("road_map_shortest_path", SHORT_PATH_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate road_map EV charging requests.")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--requests-per-day", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-long-trip-km", type=float, default=120.0)
    parser.add_argument("--long-trip-share", type=float, default=0.82)
    parser.add_argument("--safety-soc", type=float, default=0.04)
    parser.add_argument("--max-attempts-per-request", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
