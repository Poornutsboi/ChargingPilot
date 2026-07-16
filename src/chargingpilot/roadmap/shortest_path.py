from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NODES_PATH = BASE_DIR / "nodes_final.geojson"
DEFAULT_LINKS_PATH = BASE_DIR / "links_final.geojson"
DEFAULT_LINK_FLOWS_PATH = BASE_DIR / "link_24h_flows.csv"
VISUALIZE_MODULE_PATH = BASE_DIR / "visualize_road_map.py"
VDF_MODULE_PATH = BASE_DIR / "vdf.py"


@dataclass(frozen=True)
class PathResult:
    node_ids: list[int]
    distance_m: float
    travel_time_min: float | None = None


@dataclass(frozen=True)
class RoadMap:
    graph: dict[int, list[tuple[int, float]]]
    toll_ids: set[int]
    node_types: dict[int, str]
    edge_lengths_m: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_weights: dict[tuple[int, int], float] = field(default_factory=dict)
    cost_metric: str = "distance_m"


def load_road_map(
    nodes_path: Path,
    links_path: Path,
    undirected: bool = False,
    *,
    weight_mode: str = "distance",
    hour_of_day: int = 0,
    link_flows_path: Path | None = None,
    link_parameters_path: Path | None = None,
) -> RoadMap:
    if weight_mode not in {"distance", "travel_time"}:
        raise ValueError("weight_mode must be 'distance' or 'travel_time'")
    visualize = _load_visualize_module()
    network = visualize.load_network(nodes_path, links_path)
    node_types = {int(node["id"]): str(node["type"]) for node in network["nodes"] if node.get("id") is not None}
    toll_ids = {node_id for node_id, node_type in node_types.items() if node_type == "toll"}
    graph: dict[int, list[tuple[int, float]]] = {node_id: [] for node_id in node_types}
    edge_lengths_m: dict[tuple[int, int], float] = {}
    edge_weights: dict[tuple[int, int], float] = {}
    vdf = _load_vdf_module() if weight_mode == "travel_time" else None
    hourly_flows = (
        vdf.load_hourly_flows(link_flows_path or DEFAULT_LINK_FLOWS_PATH)
        if vdf is not None
        else {}
    )
    link_parameters = (
        vdf.load_link_parameters(link_parameters_path)
        if vdf is not None and link_parameters_path is not None
        else {}
    )

    for link_id, link in enumerate(network["links"], start=1):
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        length = link.get("length")
        if from_id is None or to_id is None or length is None:
            continue
        source = int(from_id)
        target = int(to_id)
        length_m = float(length)
        if vdf is None:
            weight = length_m
        else:
            parameters = link_parameters.get(
                int(link_id),
                vdf.default_link_parameters(
                    link_id=int(link_id),
                    from_id=source,
                    to_id=target,
                    length_m=length_m,
                ),
            )
            weight = float(vdf.link_travel_time_min(parameters, hourly_flows, int(hour_of_day)))
        graph.setdefault(source, []).append((target, weight))
        graph.setdefault(target, graph.get(target, []))
        edge_lengths_m[(source, target)] = length_m
        edge_weights[(source, target)] = weight
        if undirected:
            graph[target].append((source, weight))
            edge_lengths_m[(target, source)] = length_m
            edge_weights[(target, source)] = weight

    return RoadMap(
        graph=graph,
        toll_ids=toll_ids,
        node_types=node_types,
        edge_lengths_m=edge_lengths_m,
        edge_weights=edge_weights,
        cost_metric="travel_time_min" if weight_mode == "travel_time" else "distance_m",
    )


def shortest_path_between_tolls(road_map: RoadMap, origin: int, destination: int) -> PathResult:
    if origin not in road_map.toll_ids:
        raise ValueError(f"origin {origin} is not a toll node")
    if destination not in road_map.toll_ids:
        raise ValueError(f"destination {destination} is not a toll node")
    return shortest_path_between_nodes(road_map, origin, destination)


def shortest_path_between_nodes(road_map: RoadMap, origin: int, destination: int) -> PathResult:
    raw = dijkstra_shortest_path(road_map.graph, origin, destination)
    return _result_for_metric(road_map, raw.node_ids, raw.distance_m)


def shortest_path_via_charging_stations(
    road_map: RoadMap, origin: int, destination: int, charging_station_ids: list[int]
) -> PathResult:
    if not 1 <= len(charging_station_ids) <= 2:
        raise ValueError("charging_station_ids must contain 1 or 2 node ids")
    if len(set(charging_station_ids)) != len(charging_station_ids):
        raise ValueError("charging_station_ids must not contain duplicate node ids")

    points = [int(origin), *[int(node_id) for node_id in charging_station_ids], int(destination)]
    return _shortest_path_through_ordered_points(road_map, points)


def dijkstra_shortest_path(
    graph: dict[int, list[tuple[int, float]]],
    origin: int,
    destination: int,
    forbidden_nodes: set[int] | None = None,
) -> PathResult:
    if origin not in graph:
        raise ValueError(f"origin {origin} is not in the graph")
    if destination not in graph:
        raise ValueError(f"destination {destination} is not in the graph")
    forbidden = set() if forbidden_nodes is None else {int(node_id) for node_id in forbidden_nodes}
    forbidden.discard(int(origin))
    forbidden.discard(int(destination))

    distances: dict[int, float] = {origin: 0.0}
    previous: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, origin)]
    visited: set[int] = set()

    while queue:
        distance, node_id = heapq.heappop(queue)
        if node_id in visited:
            continue
        visited.add(node_id)
        if node_id in forbidden:
            continue
        if node_id == destination:
            return PathResult(node_ids=_reconstruct_path(previous, origin, destination), distance_m=distance)

        for neighbor, weight in graph.get(node_id, []):
            if neighbor in forbidden:
                continue
            if weight < 0:
                raise ValueError("Dijkstra requires non-negative edge weights")
            next_distance = distance + weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                previous[neighbor] = node_id
                heapq.heappush(queue, (next_distance, neighbor))

    raise ValueError(f"no path from {origin} to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a shortest path, optionally through 1 or 2 required charging stations."
    )
    parser.add_argument("--origin", type=int, required=True, help="Origin node id.")
    parser.add_argument("--destination", type=int, required=True, help="Destination node id.")
    parser.add_argument(
        "--charging-stations",
        type=int,
        nargs="+",
        default=[],
        help="One or two required charging/service station node ids, visited in the given order.",
    )
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS_PATH)
    parser.add_argument(
        "--vdf",
        action="store_true",
        help="Use Modified VDF travel time instead of distance as the shortest-path edge weight.",
    )
    parser.add_argument("--hour-of-day", type=int, default=0, help="Hour used for VDF hourly flow lookup.")
    parser.add_argument("--link-flows", type=Path, default=DEFAULT_LINK_FLOWS_PATH)
    parser.add_argument("--link-parameters", type=Path, default=None)
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Treat every link as bidirectional. By default GeoJSON from_id -> to_id direction is respected.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    road_map = load_road_map(
        args.nodes,
        args.links,
        undirected=args.undirected,
        weight_mode="travel_time" if args.vdf else "distance",
        hour_of_day=int(args.hour_of_day),
        link_flows_path=args.link_flows,
        link_parameters_path=args.link_parameters,
    )
    if args.charging_stations:
        result = shortest_path_via_charging_stations(
            road_map, args.origin, args.destination, args.charging_stations
        )
    else:
        result = shortest_path_between_tolls(road_map, args.origin, args.destination)
    output = {
        "origin": args.origin,
        "destination": args.destination,
        "charging_station_ids": args.charging_stations,
        "cost_metric": road_map.cost_metric,
        "distance_m": result.distance_m,
        "distance_km": result.distance_m / 1000.0,
        "travel_time_min": result.travel_time_min,
        "node_count": len(result.node_ids),
        "node_ids": result.node_ids,
        "directed": not args.undirected,
    }
    indent = 2 if args.pretty else None
    print(json.dumps(output, ensure_ascii=False, indent=indent))


def _load_visualize_module() -> Any:
    spec = importlib.util.spec_from_file_location("visualize_road_map", VISUALIZE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_vdf_module() -> Any:
    spec = importlib.util.spec_from_file_location("road_map_vdf", VDF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reconstruct_path(previous: dict[int, int], origin: int, destination: int) -> list[int]:
    node_id = destination
    path = [node_id]
    while node_id != origin:
        node_id = previous[node_id]
        path.append(node_id)
    path.reverse()
    return path


def _shortest_path_through_ordered_points(
    road_map: RoadMap, points: list[int]
) -> PathResult:
    total_distance = 0.0
    total_travel_time = 0.0 if road_map.cost_metric == "travel_time_min" else None
    combined_path: list[int] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        forbidden_nodes = set(points[index + 2 :])
        raw_segment = dijkstra_shortest_path(
            road_map.graph,
            int(left),
            int(right),
            forbidden_nodes=forbidden_nodes,
        )
        segment = _result_for_metric(road_map, raw_segment.node_ids, raw_segment.distance_m)
        if combined_path:
            combined_path.extend(segment.node_ids[1:])
        else:
            combined_path.extend(segment.node_ids)
        total_distance += segment.distance_m
        if total_travel_time is not None:
            total_travel_time += float(segment.travel_time_min or 0.0)
    return PathResult(node_ids=combined_path, distance_m=total_distance, travel_time_min=total_travel_time)


def _result_for_metric(road_map: RoadMap, node_ids: list[int], path_cost: float) -> PathResult:
    if road_map.cost_metric == "travel_time_min":
        return PathResult(
            node_ids=node_ids,
            distance_m=_sum_path_values(road_map.edge_lengths_m, node_ids, "distance"),
            travel_time_min=float(path_cost),
        )
    return PathResult(node_ids=node_ids, distance_m=float(path_cost))


def _sum_path_values(edge_values: dict[tuple[int, int], float], node_ids: list[int], label: str) -> float:
    total = 0.0
    for left, right in zip(node_ids, node_ids[1:]):
        key = (int(left), int(right))
        try:
            total += float(edge_values[key])
        except KeyError as exc:
            raise ValueError(f"missing {label} for edge {left}->{right}") from exc
    return total


if __name__ == "__main__":
    main()
