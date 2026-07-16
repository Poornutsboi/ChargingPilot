import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "chargingpilot" / "roadmap" / "shortest_path.py"


def load_module():
    spec = importlib.util.spec_from_file_location("shortest_path", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RoadMapShortestPathTests(unittest.TestCase):
    def test_dijkstra_finds_shortest_path_between_tolls(self):
        module = load_module()
        graph = {
            1: [(2, 10.0), (3, 2.0)],
            2: [(4, 2.0)],
            3: [(2, 1.0), (4, 20.0)],
            4: [],
        }

        result = module.dijkstra_shortest_path(graph, 1, 4)

        self.assertEqual(result.node_ids, [1, 3, 2, 4])
        self.assertTrue(math.isclose(result.distance_m, 5.0))

    def test_shortest_path_with_one_required_charging_station(self):
        module = load_module()
        road_map = module.RoadMap(
            graph={
                1: [(2, 2.0), (3, 1.0)],
                2: [(4, 2.0)],
                3: [(4, 1.0)],
                4: [],
            },
            toll_ids={1, 2, 3, 4},
            node_types={1: "toll", 2: "toll", 3: "toll", 4: "toll"},
        )

        result = module.shortest_path_via_charging_stations(road_map, 1, 4, [2])

        self.assertEqual(result.node_ids, [1, 2, 4])
        self.assertTrue(math.isclose(result.distance_m, 4.0))

    def test_shortest_path_with_two_required_charging_stations_keeps_given_order(self):
        module = load_module()
        road_map = module.RoadMap(
            graph={
                1: [(2, 10.0), (3, 1.0)],
                2: [(3, 1.0), (4, 1.0)],
                3: [(2, 1.0), (4, 10.0)],
                4: [],
            },
            toll_ids={1, 2, 3, 4},
            node_types={1: "toll", 2: "toll", 3: "toll", 4: "toll"},
        )

        result = module.shortest_path_via_charging_stations(road_map, 1, 4, [2, 3])

        self.assertEqual(result.node_ids, [1, 2, 3, 2, 4])
        self.assertTrue(math.isclose(result.distance_m, 13.0))

    def test_load_road_map_rejects_non_toll_endpoints(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            nodes_path = tmp_root / "nodes.geojson"
            links_path = tmp_root / "links.geojson"
            nodes_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"id": 1, "type": "toll"},
                                "geometry": {"type": "Point", "coordinates": [0, 0]},
                            },
                            {
                                "type": "Feature",
                                "properties": {"id": 2, "type": "interchange"},
                                "geometry": {"type": "Point", "coordinates": [3, 4]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            links_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"from_id": 1, "to_id": 2, "length": None},
                                "geometry": {"type": "LineString", "coordinates": [[0, 0], [3, 4]]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            road_map = module.load_road_map(nodes_path, links_path)

        with self.assertRaisesRegex(ValueError, "destination .* is not a toll"):
            module.shortest_path_between_tolls(road_map, 1, 2)

    def test_load_road_map_uses_computed_lengths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            nodes_path = tmp_root / "nodes.geojson"
            links_path = tmp_root / "links.geojson"
            nodes_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"id": 1, "type": "toll"},
                                "geometry": {"type": "Point", "coordinates": [0, 0]},
                            },
                            {
                                "type": "Feature",
                                "properties": {"id": 2, "type": "toll"},
                                "geometry": {"type": "Point", "coordinates": [3, 4]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            links_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"from_id": 1, "to_id": 2, "length": None},
                                "geometry": {"type": "LineString", "coordinates": [[0, 0], [3, 4]]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            road_map = module.load_road_map(nodes_path, links_path)
            result = module.shortest_path_between_tolls(road_map, 1, 2)

        self.assertEqual(result.node_ids, [1, 2])
        self.assertTrue(math.isclose(result.distance_m, 5.0))

    def test_vdf_weighted_road_map_uses_hourly_travel_time_as_path_cost(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            nodes_path = tmp_root / "nodes.geojson"
            links_path = tmp_root / "links.geojson"
            flows_path = tmp_root / "link_24h_flows.csv"
            parameters_path = tmp_root / "link_parameters.csv"
            nodes_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"id": node_id, "type": "toll"},
                                "geometry": {"type": "Point", "coordinates": [node_id, 0]},
                            }
                            for node_id in [1, 2, 4]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            links_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"from_id": 1, "to_id": 4, "length": 1000},
                                "geometry": {"type": "LineString", "coordinates": [[1, 0], [4, 0]]},
                            },
                            {
                                "type": "Feature",
                                "properties": {"from_id": 1, "to_id": 2, "length": 1000},
                                "geometry": {"type": "LineString", "coordinates": [[1, 0], [2, 0]]},
                            },
                            {
                                "type": "Feature",
                                "properties": {"from_id": 2, "to_id": 4, "length": 1000},
                                "geometry": {"type": "LineString", "coordinates": [[2, 0], [4, 0]]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            flows_path.write_text(
                "link_id,from_id,to_id,flow_h00,flow_h08\n"
                "1,1,4,0,1900\n"
                "2,1,2,0,0\n"
                "3,2,4,0,0\n",
                encoding="utf-8",
            )
            parameters_path.write_text(
                "link_id,from_id,to_id,length_m,lane_count,free_flow_speed_kmph,capacity_veh_per_hour,m\n"
                "1,1,4,1000,1,60,1000,2.5\n"
                "2,1,2,1000,1,60,1000,2.5\n"
                "3,2,4,1000,1,60,1000,2.5\n",
                encoding="utf-8",
            )

            road_map = module.load_road_map(
                nodes_path,
                links_path,
                weight_mode="travel_time",
                hour_of_day=8,
                link_flows_path=flows_path,
                link_parameters_path=parameters_path,
            )
            result = module.shortest_path_between_tolls(road_map, 1, 4)

        self.assertEqual(result.node_ids, [1, 2, 4])
        self.assertTrue(math.isclose(result.distance_m, 2000.0))
        self.assertIsNotNone(result.travel_time_min)
        self.assertGreater(result.travel_time_min, 0.0)


if __name__ == "__main__":
    unittest.main()
