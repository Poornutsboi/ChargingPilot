from __future__ import annotations

import heapq
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .cache import CacheRecord, build_cache_key, load_cache, write_cache
from .errors import NoMandatoryServiceRouteError, RouteCacheError
from .models import RouteLeg, RouteResult, ServiceBaseline


def _load_shortest_path_module():
    from chargingpilot.roadmap import shortest_path

    return shortest_path


class RoadDistanceOracle:
    def __init__(
        self,
        road_map: Any,
        service_ids: tuple[int, ...],
        *,
        directed: bool,
        cache_path: Path | None = None,
    ) -> None:
        if road_map.cost_metric != "distance_m":
            raise ValueError(
                "RoadDistanceOracle requires road_map.cost_metric == 'distance_m'; "
                f"got {road_map.cost_metric!r}"
            )
        station_ids = tuple(int(node_id) for node_id in service_ids)
        if not station_ids or any(
            left >= right for left, right in zip(station_ids, station_ids[1:])
        ):
            raise ValueError("service_ids must be strictly ascending and unique")

        graph_nodes = set(int(node_id) for node_id in road_map.graph)
        graph_nodes.update(
            int(target)
            for neighbors in road_map.graph.values()
            for target, _ in neighbors
        )
        if not set(station_ids).issubset(graph_nodes):
            missing = sorted(set(station_ids) - graph_nodes)
            raise ValueError(f"service_ids are not graph nodes: {missing}")

        self._road_map = road_map
        self.directed = bool(directed)
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.node_ids = tuple(sorted(graph_nodes))
        self.station_ids = station_ids
        self.node_to_index = MappingProxyType(
            {node_id: index for index, node_id in enumerate(self.node_ids)}
        )
        self.station_to_index = MappingProxyType(
            {node_id: index for index, node_id in enumerate(self.station_ids)}
        )
        self.distances_m, self.next_hops = self._build_all_node_tables()
        finite = [
            distance
            for row in self.distances_m
            for distance in row
            if math.isfinite(distance)
        ]
        self.max_finite_distance_m = max(finite, default=0.0)
        self._cache_key = build_cache_key(
            road_map,
            self.node_ids,
            directed=self.directed,
            station_ids=self.station_ids,
        )
        self._cache_records: dict[tuple[int, int], CacheRecord] = {}
        self._baseline_results: dict[tuple[int, int], ServiceBaseline] = {}
        self._load_persisted_cache()

    def service_baseline(self, origin: int, destination: int) -> ServiceBaseline:
        od = (int(origin), int(destination))
        existing = self._baseline_results.get(od)
        if existing is not None:
            return existing

        result = self._product_dijkstra(*od)
        cached = self._cache_records.get(od)
        if cached is not None and (
            cached.station_id != result.station_id
            or not math.isclose(
                cached.distance_m,
                result.distance_m,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise RouteCacheError(
                self.cache_path or Path("<memory>"),
                f"cached metadata disagrees with restored route for {od}",
            )

        self._baseline_results[od] = result
        if cached is None:
            self._cache_records[od] = self._record_for(od, result)
            self._write_persisted_cache()
        return result

    def route_via(
        self,
        origin: int,
        destination: int,
        station_ids: tuple[int, ...],
    ) -> RouteResult:
        required = tuple(int(node_id) for node_id in station_ids)
        shortest_path = _load_shortest_path_module()
        raw = shortest_path.shortest_path_via_charging_stations(
            self._road_map,
            int(origin),
            int(destination),
            list(required),
        )
        return self._route_result(int(origin), int(destination), required, raw)

    def direct_route(self, origin: int, destination: int) -> RouteResult:
        shortest_path = _load_shortest_path_module()
        raw = shortest_path.shortest_path_between_nodes(
            self._road_map, int(origin), int(destination)
        )
        return self._route_result(int(origin), int(destination), (), raw)

    def _product_dijkstra(self, origin: int, destination: int) -> ServiceBaseline:
        if origin not in self.node_to_index or destination not in self.node_to_index:
            raise NoMandatoryServiceRouteError(origin, destination)
        services = set(self.station_ids)
        initial_path = (int(origin),)
        initial_state = (int(origin), int(origin) in services)
        best: dict[tuple[int, bool], tuple[float, tuple[int, ...]]] = {
            initial_state: (0.0, initial_path)
        }
        queue: list[
            tuple[
                float,
                tuple[int, ...],
                int,
                bool,
                tuple[tuple[int, bool], ...],
            ]
        ] = [
            (0.0, initial_path, initial_state[0], initial_state[1], (initial_state,))
        ]

        while queue:
            distance, path, node_id, visited_service, state_path = heapq.heappop(queue)
            state = (node_id, visited_service)
            if best.get(state) != (distance, path):
                continue
            if node_id == destination and visited_service:
                station_id = next(node for node in path if node in services)
                return ServiceBaseline(station_id, distance, path)
            for neighbor, weight in sorted(self._road_map.graph.get(node_id, ())):
                neighbor = int(neighbor)
                weight = float(weight)
                if weight < 0:
                    raise ValueError("Dijkstra requires non-negative edge weights")
                next_distance = distance + weight
                next_path = (*path, neighbor)
                next_state = (neighbor, visited_service or neighbor in services)
                if next_state in state_path:
                    continue
                candidate = (next_distance, next_path)
                if candidate < best.get(next_state, (math.inf, ())):
                    best[next_state] = candidate
                    heapq.heappush(
                        queue,
                        (
                            next_distance,
                            next_path,
                            neighbor,
                            next_state[1],
                            (*state_path, next_state),
                        ),
                    )
        raise NoMandatoryServiceRouteError(origin, destination)

    def _single_source(
        self, source: int
    ) -> tuple[dict[int, float], dict[int, int | None]]:
        best: dict[int, tuple[float, tuple[int, ...]]] = {source: (0.0, (source,))}
        queue: list[tuple[float, tuple[int, ...], int]] = [(0.0, (source,), source)]
        while queue:
            distance, path, node_id = heapq.heappop(queue)
            if best.get(node_id) != (distance, path):
                continue
            for neighbor, weight in sorted(self._road_map.graph.get(node_id, ())):
                neighbor = int(neighbor)
                weight = float(weight)
                if weight < 0:
                    raise ValueError("Dijkstra requires non-negative edge weights")
                if neighbor in path:
                    continue
                candidate = (distance + weight, (*path, neighbor))
                if candidate < best.get(neighbor, (math.inf, ())):
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
        distances = {
            node_id: best.get(node_id, (math.inf, ()))[0]
            for node_id in self.node_ids
        }
        next_hops = {
            node_id: (
                source
                if node_id == source
                else best[node_id][1][1]
                if node_id in best
                else None
            )
            for node_id in self.node_ids
        }
        return distances, next_hops

    def _build_all_node_tables(
        self,
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[int | None, ...], ...]]:
        distance_rows = []
        next_hop_rows = []
        for source in self.node_ids:
            distances, next_hops = self._single_source(source)
            distance_rows.append(tuple(distances[target] for target in self.node_ids))
            next_hop_rows.append(tuple(next_hops[target] for target in self.node_ids))
        return tuple(distance_rows), tuple(next_hop_rows)

    def _route_result(
        self,
        origin: int,
        destination: int,
        required: tuple[int, ...],
        raw: Any,
    ) -> RouteResult:
        path = tuple(int(node_id) for node_id in raw.node_ids)
        points = (origin, *required, destination)
        legs = []
        start = 0
        for source, target in zip(points, points[1:]):
            try:
                end = path.index(target, start)
            except ValueError as exc:
                raise ValueError(
                    f"combined route does not contain required point {target}"
                ) from exc
            leg_nodes = path[start : end + 1]
            legs.append(RouteLeg(source, target, leg_nodes, self._path_distance(leg_nodes)))
            start = end
        leg_total = sum(leg.distance_m for leg in legs)
        if not math.isclose(
            leg_total,
            float(raw.distance_m),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"route leg distance {leg_total} does not match combined distance {raw.distance_m}"
            )
        return RouteResult(required, tuple(legs), path, float(raw.distance_m))

    def _path_distance(self, path: tuple[int, ...]) -> float:
        total = 0.0
        for source, target in zip(path, path[1:]):
            edge = (source, target)
            if edge in self._road_map.edge_lengths_m:
                total += float(self._road_map.edge_lengths_m[edge])
                continue
            weights = [
                float(weight)
                for neighbor, weight in self._road_map.graph.get(source, ())
                if int(neighbor) == target
            ]
            if not weights:
                raise ValueError(f"missing distance for edge {source}->{target}")
            total += min(weights)
        return total

    def _load_persisted_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            persisted_key, records = load_cache(self.cache_path)
            if persisted_key == self._cache_key:
                self._cache_records = {
                    (record.origin, record.destination): record for record in records
                }
                return
            rebuilt = {}
            for record in records:
                od = (record.origin, record.destination)
                baseline = self._product_dijkstra(*od)
                rebuilt[od] = self._record_for(od, baseline)
                self._baseline_results[od] = baseline
            self._cache_records = rebuilt
            self._write_persisted_cache()
        except RouteCacheError:
            raise
        except Exception as exc:
            raise RouteCacheError(self.cache_path, str(exc)) from exc

    def _write_persisted_cache(self) -> None:
        if self.cache_path is None:
            return
        try:
            write_cache(
                self.cache_path,
                self._cache_key,
                tuple(self._cache_records.values()),
            )
        except Exception as exc:
            raise RouteCacheError(self.cache_path, str(exc)) from exc

    @staticmethod
    def _record_for(
        od: tuple[int, int], baseline: ServiceBaseline
    ) -> CacheRecord:
        return CacheRecord(od[0], od[1], baseline.station_id, baseline.distance_m)
