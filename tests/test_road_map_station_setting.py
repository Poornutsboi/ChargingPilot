import csv
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path

import yaml

from chargingpilot.environment.data_factory import DataFactory
from chargingpilot.roadmap.generate_link_24h_flow import generate_link_flows


SETTING_PATH = Path(__file__).resolve().parents[1] / "configs" / "setting_72stations_roadmap_pv_ess.yaml"
ROAD_MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "roadmap"
CITY_TIERS = {
    "Shanghai": 1.0, "Suzhou": 0.9, "Hangzhou": 0.9, "Jiaxing": 0.7,
    "Ningbo": 0.55, "Wuxi": 0.55, "Shaoxing": 0.4, "Huzhou": 0.25,
}
NON_RENEWABLE_SERVICE_IDS = {31, 317, 342, 344, 354, 362, 524, 569, 643, 645}


def _half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _interpolate(rank: int, start_rank: int, end_rank: int, low: int, high: int) -> int:
    return _half_up(low + (rank - start_rank) * (high - low) / (end_rank - start_rank))


def _expected_station_capacities() -> dict[int, int]:
    nodes = json.loads((ROAD_MAP_PATH / "nodes_final.geojson").read_text(encoding="utf-8"))
    service_ids = sorted(
        int(feature["properties"]["id"])
        for feature in nodes["features"]
        if feature["properties"]["type"] == "service"
    )
    flows = defaultdict(int)
    city_scores = defaultdict(float)
    for link in generate_link_flows(ROAD_MAP_PATH / "nodes_final.geojson", ROAD_MAP_PATH / "links_final.geojson"):
        daily_flow = int(link["daily_flow"])
        city_tier = CITY_TIERS[link["nearest_city"]]
        for node_id in (int(link["from_id"]), int(link["to_id"])):
            flows[node_id] += daily_flow
            city_scores[node_id] = max(city_scores[node_id], city_tier)

    min_flow = min(flows[node_id] for node_id in service_ids)
    max_flow = max(flows[node_id] for node_id in service_ids)
    ranked_ids = sorted(
        service_ids,
        key=lambda node_id: (
            0.75 * (flows[node_id] - min_flow) / (max_flow - min_flow)
            + 0.25 * city_scores[node_id],
            node_id,
        ),
    )
    capacities = {}
    for rank, node_id in enumerate(ranked_ids, start=1):
        if rank <= 18:
            capacities[node_id] = _interpolate(rank, 1, 18, 8, 12)
        elif rank <= 54:
            capacities[node_id] = _interpolate(rank, 19, 54, 14, 20)
        else:
            capacities[node_id] = _interpolate(rank, 55, 72, 22, 30)
    return {node_id: capacities[node_id] for node_id in service_ids}


def _expected_renewable_station_ids() -> set[int]:
    nodes = json.loads((ROAD_MAP_PATH / "nodes_final.geojson").read_text(encoding="utf-8"))
    service_ids = sorted(
        int(feature["properties"]["id"])
        for feature in nodes["features"]
        if feature["properties"]["type"] == "service"
    )
    flows = defaultdict(int)
    for link in generate_link_flows(ROAD_MAP_PATH / "nodes_final.geojson", ROAD_MAP_PATH / "links_final.geojson"):
        for node_id in (int(link["from_id"]), int(link["to_id"])):
            if node_id in service_ids:
                flows[node_id] += int(link["daily_flow"])
    base_ids = sorted(service_ids, key=lambda node_id: (flows[node_id], node_id))[:22]
    return set(base_ids) - NON_RENEWABLE_SERVICE_IDS


class RoadMapStationSettingTests(unittest.TestCase):
    def test_road_map_station_setting_matches_service_capacity_contract(self) -> None:
        settings = DataFactory._load_settings(SETTING_PATH)
        with SETTING_PATH.open(encoding="utf-8") as yaml_file:
            raw_settings = yaml.safe_load(yaml_file)

        station_ids = [int(item) for item in settings["station_ids"]]
        capacities = [int(item) for item in settings["stations"]["charge_capacity"]]
        p_max_kw = [float(item) for item in settings["stations"]["p_max_kw"]]
        p_grid_max_kw = [float(item) for item in settings["stations"]["p_grid_max_kw"]]
        pv_indicator = [int(item) for item in raw_settings["renewable"]["pv_indicator"]]
        ess_indicator = [int(item) for item in raw_settings["ess"]["ess_indicator"]]
        expected_capacities = _expected_station_capacities()
        expected_renewable_ids = _expected_renewable_station_ids()

        self.assertEqual(station_ids, list(expected_capacities))
        self.assertEqual(dict(zip(station_ids, capacities, strict=True)), expected_capacities)
        self.assertEqual(p_max_kw, [capacity * 108.0 for capacity in capacities])
        self.assertEqual(p_grid_max_kw, [capacity * 108.0 for capacity in capacities])
        self.assertEqual(pv_indicator, ess_indicator)
        self.assertEqual(sum(pv_indicator), 17)
        self.assertEqual(
            {station_id for station_id, enabled in zip(station_ids, pv_indicator, strict=True) if enabled},
            expected_renewable_ids,
        )
        self.assertEqual(settings["environment"]["max_station_count"], 72)


if __name__ == "__main__":
    unittest.main()
