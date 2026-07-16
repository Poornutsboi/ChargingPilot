from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "chargingpilot"
    / "roadmap"
    / "shortest_path.py"
)


def load_shortest_path_module():
    spec = importlib.util.spec_from_file_location("distance_oracle_shortest_path", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shortest_path = load_shortest_path_module()


def road_map(edges: list[tuple[int, int, float]], node_ids: tuple[int, ...] | None = None):
    nodes = set(node_ids or ())
    nodes.update(source for source, _, _ in edges)
    nodes.update(target for _, target, _ in edges)
    graph = {node_id: [] for node_id in sorted(nodes)}
    edge_values = {}
    for source, target, distance_m in edges:
        graph[source].append((target, distance_m))
        edge_values[(source, target)] = distance_m
    return shortest_path.RoadMap(
        graph=graph,
        toll_ids=set(nodes),
        node_types={node_id: "toll" for node_id in nodes},
        edge_lengths_m=dict(edge_values),
        edge_weights=dict(edge_values),
    )


class RoadDistanceOracleTests(TestCase):
    def test_constructor_rejects_travel_time_cost_map(self):
        from chargingpilot.routing import RoadDistanceOracle

        travel_time_map = shortest_path.RoadMap(
            graph={1: [(2, 1.5)], 2: [(4, 2.5)], 4: []},
            toll_ids={1, 2, 4},
            node_types={1: "toll", 2: "toll", 4: "toll"},
            edge_lengths_m={(1, 2): 1_000.0, (2, 4): 2_000.0},
            edge_weights={(1, 2): 1.5, (2, 4): 2.5},
            cost_metric="travel_time_min",
        )

        with self.assertRaisesRegex(ValueError, "distance_m"):
            RoadDistanceOracle(travel_time_map, (2,), directed=True)

    def test_baseline_requires_service(self):
        from chargingpilot.routing import RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map([(1, 4, 2.0), (1, 2, 2.0), (2, 4, 2.0)]),
            (2,),
            directed=True,
        )

        result = oracle.service_baseline(1, 4)

        self.assertEqual(result.node_ids, (1, 2, 4))
        self.assertEqual(result.station_id, 2)
        self.assertEqual(result.distance_m, 4.0)

        return_oracle = RoadDistanceOracle(
            road_map([(1, 2, 1.0), (2, 1, 1.0), (1, 4, 1.0)]),
            (2,),
            directed=True,
        )
        return_result = return_oracle.service_baseline(1, 4)
        self.assertEqual(return_result.node_ids, (1, 2, 1, 4))

    def test_baseline_uses_first_traversed_service(self):
        from chargingpilot.routing import RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map([(1, 3, 1.0), (3, 2, 1.0), (2, 4, 1.0)]),
            (2, 3),
            directed=True,
        )

        result = oracle.service_baseline(1, 4)

        self.assertEqual(result.node_ids, (1, 3, 2, 4))
        self.assertEqual(result.station_id, 3)

    def test_baseline_tie_is_lexicographic(self):
        from chargingpilot.routing import RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map(
                [(1, 3, 1.0), (3, 4, 1.0), (1, 2, 1.0), (2, 4, 1.0)]
            ),
            (2, 3),
            directed=True,
        )

        result = oracle.service_baseline(1, 4)

        self.assertEqual(result.node_ids, (1, 2, 4))
        self.assertEqual(result.station_id, 2)

    def test_origin_service_is_valid(self):
        from chargingpilot.routing import RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map([(1, 4, 3.0)]),
            (1,),
            directed=True,
        )

        result = oracle.service_baseline(1, 4)

        self.assertEqual(result.station_id, 1)
        self.assertEqual(result.node_ids, (1, 4))
        self.assertEqual(result.distance_m, 3.0)

    def test_no_service_route_has_od_context(self):
        from chargingpilot.routing import NoMandatoryServiceRouteError, RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map([(1, 4, 1.0)], node_ids=(1, 2, 4)),
            (2,),
            directed=True,
        )

        with self.assertRaises(NoMandatoryServiceRouteError) as caught:
            oracle.service_baseline(1, 4)

        self.assertEqual(caught.exception.origin, 1)
        self.assertEqual(caught.exception.destination, 4)
        self.assertIn("1", str(caught.exception))
        self.assertIn("4", str(caught.exception))

    def test_route_via_has_exact_legs(self):
        from chargingpilot.routing import RouteLeg, RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map(
                [
                    (1, 5, 1.0),
                    (5, 2, 2.0),
                    (2, 6, 3.0),
                    (6, 3, 4.0),
                    (3, 4, 5.0),
                ]
            ),
            (2, 3),
            directed=True,
        )

        result = oracle.route_via(1, 4, (2, 3))

        self.assertEqual(result.required_station_ids, (2, 3))
        self.assertEqual(result.node_ids, (1, 5, 2, 6, 3, 4))
        self.assertEqual(
            result.legs,
            (
                RouteLeg(1, 2, (1, 5, 2), 3.0),
                RouteLeg(2, 3, (2, 6, 3), 7.0),
                RouteLeg(3, 4, (3, 4), 5.0),
            ),
        )
        self.assertEqual(sum(leg.distance_m for leg in result.legs), result.distance_m)
        self.assertEqual(result.distance_m, 15.0)

        inconsistent = road_map([(1, 2, 1_000_000_000.0), (2, 4, 1_000_000_000.0)])
        inconsistent.edge_lengths_m[(1, 2)] += 0.001
        inconsistent_oracle = RoadDistanceOracle(inconsistent, (2,), directed=True)
        with self.assertRaisesRegex(ValueError, "does not match combined distance"):
            inconsistent_oracle.route_via(1, 4, (2,))

    def test_ordered_route_does_not_visit_future_waypoint(self):
        from chargingpilot.routing import RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map(
                [
                    (1, 3, 1.0),
                    (3, 2, 1.0),
                    (1, 5, 2.0),
                    (5, 2, 2.0),
                    (2, 3, 1.0),
                    (3, 4, 1.0),
                ]
            ),
            (2, 3),
            directed=True,
        )

        result = oracle.route_via(1, 4, (2, 3))

        self.assertEqual(result.legs[0].node_ids, (1, 5, 2))
        self.assertNotIn(3, result.legs[0].node_ids)
        self.assertEqual(result.node_ids, (1, 5, 2, 3, 4))

    def test_all_node_distance_and_next_hop(self):
        from chargingpilot.routing import RoadDistanceOracle

        oracle = RoadDistanceOracle(
            road_map(
                [
                    (1, 3, 1.0),
                    (1, 2, 2.0),
                    (3, 2, 1.0),
                    (2, 4, 3.0),
                    (3, 4, 10.0),
                ]
            ),
            (2, 3),
            directed=True,
        )

        one = oracle.node_to_index[1]
        two = oracle.node_to_index[2]
        four = oracle.node_to_index[4]
        self.assertEqual(oracle.node_ids, (1, 2, 3, 4))
        self.assertEqual(oracle.station_ids, (2, 3))
        self.assertEqual(oracle.station_to_index[3], 1)
        self.assertEqual(oracle.distances_m[one][two], 2.0)
        self.assertEqual(oracle.next_hops[one][two], 2)
        self.assertEqual(oracle.distances_m[one][four], 5.0)
        self.assertEqual(oracle.next_hops[one][four], 2)
        self.assertTrue(math.isinf(oracle.distances_m[four][one]))
        self.assertIsNone(oracle.next_hops[four][one])
        self.assertEqual(oracle.max_finite_distance_m, 5.0)
        self.assertIsInstance(oracle.distances_m, tuple)
        with self.assertRaises(TypeError):
            oracle.node_to_index[9] = 9

    def test_cache_key_mismatch_rebuilds(self):
        from chargingpilot.routing import RoadDistanceOracle

        graph = road_map([(1, 2, 1.0), (2, 4, 1.0)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "routes.json"
            first = RoadDistanceOracle(graph, (2,), directed=True, cache_path=cache_path)
            first.service_baseline(1, 4)
            old_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            old_payload["entries"][0]["distance_m"] = 999.0
            cache_path.write_text(json.dumps(old_payload), encoding="utf-8")

            second = RoadDistanceOracle(graph, (2,), directed=False, cache_path=cache_path)
            result = second.service_baseline(1, 4)
            new_payload = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertNotEqual(old_payload["key"], new_payload["key"])
        self.assertEqual(result.distance_m, 2.0)
        self.assertEqual(new_payload["entries"][0]["distance_m"], 2.0)

    def test_corrupt_cache_has_path_context(self):
        from chargingpilot.routing import RoadDistanceOracle, RouteCacheError

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "routes.json"
            cache_path.write_text("not json", encoding="utf-8")

            with self.assertRaises(RouteCacheError) as caught:
                RoadDistanceOracle(
                    road_map([(1, 2, 1.0), (2, 4, 1.0)]),
                    (2,),
                    directed=True,
                    cache_path=cache_path,
                )

        self.assertEqual(caught.exception.cache_path, cache_path)
        self.assertIn(str(cache_path), str(caught.exception))

    def test_failed_cache_rebuild_preserves_reason(self):
        from chargingpilot.routing import RoadDistanceOracle, RouteCacheError

        graph = road_map([(1, 2, 1.0), (2, 4, 1.0)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "routes.json"
            first = RoadDistanceOracle(graph, (2,), directed=True, cache_path=cache_path)
            first.service_baseline(1, 4)

            with mock.patch.object(
                RoadDistanceOracle,
                "_product_dijkstra",
                side_effect=RuntimeError("rebuild exploded"),
            ):
                with self.assertRaises(RouteCacheError) as caught:
                    RoadDistanceOracle(
                        graph,
                        (2,),
                        directed=False,
                        cache_path=cache_path,
                    )

        self.assertEqual(caught.exception.cache_path, cache_path)
        self.assertIn("rebuild exploded", caught.exception.reason)


if __name__ == "__main__":
    import unittest

    unittest.main()
