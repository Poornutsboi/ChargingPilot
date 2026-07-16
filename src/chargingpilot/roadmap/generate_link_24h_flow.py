from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NODES_PATH = BASE_DIR / "nodes_final.geojson"
DEFAULT_LINKS_PATH = BASE_DIR / "links_final.geojson"
DEFAULT_OUTPUT_PATH = BASE_DIR / "link_24h_flows.csv"
VISUALIZE_MODULE_PATH = BASE_DIR / "visualize_road_map.py"
HOUR_COLUMNS = [f"flow_h{hour:02d}" for hour in range(24)]


@dataclass(frozen=True)
class CityAnchor:
    name: str
    lon: float
    lat: float
    weight: float
    radius_km: float
    ring_inner_km: float
    ring_outer_km: float


@dataclass(frozen=True)
class RegionSignal:
    region: str
    nearest_city: str
    nearest_city_distance_km: float
    shanghai_distance_km: float
    score: float


CITY_ANCHORS = (
    CityAnchor("Shanghai", 121.4737, 31.2304, 2.55, 72.0, 12.0, 42.0),
    CityAnchor("Suzhou", 120.5853, 31.2989, 1.95, 52.0, 10.0, 34.0),
    CityAnchor("Hangzhou", 120.1551, 30.2741, 2.05, 58.0, 12.0, 40.0),
    CityAnchor("Ningbo", 121.5503, 29.8746, 1.80, 54.0, 10.0, 36.0),
    CityAnchor("Jiaxing", 120.7555, 30.7461, 1.35, 46.0, 9.0, 30.0),
    CityAnchor("Shaoxing", 120.5821, 30.0515, 1.20, 42.0, 8.0, 28.0),
    CityAnchor("Huzhou", 120.0868, 30.8943, 1.05, 42.0, 8.0, 26.0),
    CityAnchor("Wuxi", 120.3119, 31.4912, 1.25, 42.0, 8.0, 28.0),
)

CORRIDORS_TO_SHANGHAI = (
    ("hangzhou_shanghai", "Hangzhou", "Shanghai", 1.05, 26.0),
    ("suzhou_shanghai", "Suzhou", "Shanghai", 1.25, 22.0),
    ("ningbo_shanghai", "Ningbo", "Shanghai", 0.90, 32.0),
    ("jiaxing_shanghai", "Jiaxing", "Shanghai", 1.10, 24.0),
)


def classify_region(lon: float, lat: float) -> RegionSignal:
    distances = [(city, haversine_km(lon, lat, city.lon, city.lat)) for city in CITY_ANCHORS]
    nearest_city, nearest_distance = min(distances, key=lambda item: item[1])
    shanghai = next(city for city in CITY_ANCHORS if city.name == "Shanghai")
    shanghai_distance = haversine_km(lon, lat, shanghai.lon, shanghai.lat)

    city_score = 0.78
    ring_bonus = 0.0
    for city, distance in distances:
        city_score += city.weight * math.exp(-0.5 * (distance / city.radius_km) ** 2)
        if city.ring_inner_km <= distance <= city.ring_outer_km:
            ring_bonus += 0.42 * city.weight

    corridor_bonus = _shanghai_corridor_bonus(lon, lat)
    shanghai_direction_bonus = 0.72 * math.exp(-shanghai_distance / 135.0)
    score = city_score + ring_bonus + corridor_bonus + shanghai_direction_bonus

    if nearest_distance <= nearest_city.ring_outer_km:
        region = f"{nearest_city.name.lower()}_metro_or_ring"
    elif corridor_bonus >= 0.75:
        region = "shanghai_corridor"
    elif nearest_distance <= 70.0:
        region = f"{nearest_city.name.lower()}_regional"
    else:
        region = "peripheral"

    return RegionSignal(
        region=region,
        nearest_city=nearest_city.name,
        nearest_city_distance_km=nearest_distance,
        shanghai_distance_km=shanghai_distance,
        score=score,
    )


def hourly_profile() -> list[float]:
    values = []
    for hour in range(24):
        morning = _gaussian(hour, center=8.0, width=1.65)
        evening = _gaussian(hour, center=18.0, width=2.05)
        midday = _gaussian(hour, center=13.0, width=3.4)
        night = _gaussian(hour, center=3.0, width=2.6)
        value = 0.72 + 0.42 * morning + 0.58 * evening + 0.16 * midday - 0.24 * night
        values.append(max(0.34, value))
    mean_value = sum(values) / len(values)
    return [value / mean_value for value in values]


def propagate_connectivity_scores(links: list[dict[str, Any]], iterations: int = 4) -> dict[int, float]:
    scores = {int(link["link_id"]): float(link["initial_score"]) for link in links}
    upstream, downstream = build_connectivity(links)

    for _ in range(int(iterations)):
        next_scores: dict[int, float] = {}
        for link in links:
            link_id = int(link["link_id"])
            upstream_scores = [scores[neighbor] for neighbor in upstream[link_id]]
            downstream_scores = [scores[neighbor] for neighbor in downstream[link_id]]
            upstream_mean = sum(upstream_scores) / len(upstream_scores) if upstream_scores else scores[link_id]
            downstream_mean = sum(downstream_scores) / len(downstream_scores) if downstream_scores else scores[link_id]
            next_scores[link_id] = 0.68 * scores[link_id] + 0.21 * upstream_mean + 0.11 * downstream_mean
        scores = next_scores
    return scores


def build_connectivity(links: list[dict[str, Any]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    incoming_by_node: dict[int, list[int]] = {}
    outgoing_by_node: dict[int, list[int]] = {}
    for link in links:
        link_id = int(link["link_id"])
        from_id = int(link["from_id"])
        to_id = int(link["to_id"])
        outgoing_by_node.setdefault(from_id, []).append(link_id)
        incoming_by_node.setdefault(to_id, []).append(link_id)

    upstream: dict[int, list[int]] = {}
    downstream: dict[int, list[int]] = {}
    for link in links:
        link_id = int(link["link_id"])
        from_id = int(link["from_id"])
        to_id = int(link["to_id"])
        upstream[link_id] = [neighbor for neighbor in incoming_by_node.get(from_id, []) if neighbor != link_id]
        downstream[link_id] = [neighbor for neighbor in outgoing_by_node.get(to_id, []) if neighbor != link_id]
    return upstream, downstream


def generate_link_flows(nodes_path: Path, links_path: Path) -> list[dict[str, Any]]:
    visualize = _load_visualize_module()
    network = visualize.load_network(nodes_path, links_path)
    links = _build_link_records(network, visualize)
    propagated = propagate_connectivity_scores(links)
    upstream, downstream = build_connectivity(links)
    profile = hourly_profile()

    rows: list[dict[str, Any]] = []
    for link in links:
        link_id = int(link["link_id"])
        score = propagated[link_id]
        length_factor = _clamp(math.log1p(float(link["length_m"]) / 1000.0) / 2.6, 0.50, 1.45)
        stable_noise = 0.93 + 0.14 * _stable_unit_noise(link_id)
        avg_hourly_flow = (360.0 + 520.0 * score) * length_factor * stable_noise
        row = {
            "link_id": link_id,
            "from_id": int(link["from_id"]),
            "to_id": int(link["to_id"]),
            "length_m": round(float(link["length_m"]), 3),
            "centroid_lon": round(float(link["centroid_lon"]), 6),
            "centroid_lat": round(float(link["centroid_lat"]), 6),
            "region": link["region"],
            "nearest_city": link["nearest_city"],
            "nearest_city_distance_km": round(float(link["nearest_city_distance_km"]), 3),
            "shanghai_distance_km": round(float(link["shanghai_distance_km"]), 3),
            "initial_score": round(float(link["initial_score"]), 6),
            "propagated_score": round(float(score), 6),
            "upstream_link_count": len(upstream[link_id]),
            "downstream_link_count": len(downstream[link_id]),
            "daily_flow": 0,
        }
        daily_flow = 0
        for hour, multiplier in enumerate(profile):
            hourly = int(round(avg_hourly_flow * multiplier))
            row[f"flow_h{hour:02d}"] = max(80, hourly)
            daily_flow += row[f"flow_h{hour:02d}"]
        row["daily_flow"] = daily_flow
        rows.append(row)
    return rows


def write_link_flows(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "link_id",
        "from_id",
        "to_id",
        "length_m",
        "centroid_lon",
        "centroid_lat",
        "region",
        "nearest_city",
        "nearest_city_distance_km",
        "shanghai_distance_km",
        "initial_score",
        "propagated_score",
        "upstream_link_count",
        "downstream_link_count",
        "daily_flow",
        *HOUR_COLUMNS,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heuristic 24h traffic flow for every road_map link.")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    rows = generate_link_flows(args.nodes, args.links)
    output_path = write_link_flows(rows, args.output)
    daily_values = [int(row["daily_flow"]) for row in rows]
    print(f"wrote: {output_path}")
    print(f"links: {len(rows)}")
    print(f"daily_flow_min: {min(daily_values)}")
    print(f"daily_flow_mean: {sum(daily_values) / len(daily_values):.1f}")
    print(f"daily_flow_max: {max(daily_values)}")


def _build_link_records(network: dict[str, Any], visualize: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, link in enumerate(network["links"], start=1):
        lonlats = [visualize.web_mercator_to_lonlat(x, y) for x, y in link["points"]]
        centroid_lon = sum(lon for lon, _lat in lonlats) / len(lonlats)
        centroid_lat = sum(lat for _lon, lat in lonlats) / len(lonlats)
        signal = classify_region(centroid_lon, centroid_lat)
        records.append(
            {
                "link_id": index,
                "from_id": int(link["from_id"]),
                "to_id": int(link["to_id"]),
                "length_m": float(link["length"]),
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "region": signal.region,
                "nearest_city": signal.nearest_city,
                "nearest_city_distance_km": signal.nearest_city_distance_km,
                "shanghai_distance_km": signal.shanghai_distance_km,
                "initial_score": signal.score,
            }
        )
    return records


def _load_visualize_module() -> Any:
    spec = importlib.util.spec_from_file_location("visualize_road_map", VISUALIZE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _shanghai_corridor_bonus(lon: float, lat: float) -> float:
    city_by_name = {city.name: city for city in CITY_ANCHORS}
    bonus = 0.0
    for _name, start_name, end_name, weight, width_km in CORRIDORS_TO_SHANGHAI:
        start = city_by_name[start_name]
        end = city_by_name[end_name]
        distance = _point_to_segment_distance_km(lon, lat, start.lon, start.lat, end.lon, end.lat)
        bonus += weight * math.exp(-0.5 * (distance / width_km) ** 2)
    return bonus


def _point_to_segment_distance_km(
    lon: float, lat: float, lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    mean_lat = math.radians((lat + lat1 + lat2) / 3.0)

    def project(point_lon: float, point_lat: float) -> tuple[float, float]:
        x = (point_lon - lon1) * 111.320 * math.cos(mean_lat)
        y = (point_lat - lat1) * 110.574
        return x, y

    px, py = project(lon, lat)
    ax, ay = 0.0, 0.0
    bx, by = project(lon2, lat2)
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = _clamp(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _gaussian(value: float, *, center: float, width: float) -> float:
    return math.exp(-0.5 * ((float(value) - float(center)) / float(width)) ** 2)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _stable_unit_noise(value: int) -> float:
    raw = math.sin(int(value) * 12.9898 + 78.233) * 43758.5453
    return raw - math.floor(raw)


if __name__ == "__main__":
    main()
