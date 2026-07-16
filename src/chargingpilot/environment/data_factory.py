from __future__ import annotations

import csv
import importlib.util
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
import yaml

from chargingpilot.environment.models import EpisodeData, SplitChargingEnvConfig, VehicleRequest
from chargingpilot.environment.request_manifest import RequestManifest
from chargingpilot.network.GCT import (
    GraphConvolutionalTransformer,
    TravelTimeBatch,
    TravelTimeModelConfig,
    predict_travel_time,
)
from chargingpilot.network.highway_travel_time import (
    HighwayTravelTimeScenario,
    build_default_highway_scenario,
)
from chargingpilot.simulator.models import StationSpec, VehicleSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHORT_PATH_MODULE_PATH = PROJECT_ROOT / "roadmap" / "shortest_path.py"


@dataclass(frozen=True)
class DataFactoryConfig:
    episodes_dir: str | Path = Path("datasets/train")
    station_setting_path: str | Path = Path("exps/data/setting_7stations_pv_ess.yaml")
    request_manifest_path: Path | None = None
    request_split: Literal["train", "validation", "test"] = "train"
    seed: int | None = None
    shuffle: bool = True
    vehicle_p_max_kw: float = 180.0
    vehicle_p_min_kw: float = 30.0
    vehicle_soc_min: float = 0.0
    travel_time_model_path: str | Path | None = None
    use_vdf_road_map: bool = False
    road_map_nodes_path: str | Path | None = None
    road_map_links_path: str | Path | None = None
    link_flows_path: str | Path | None = None
    link_parameters_path: str | Path | None = None
    road_map_undirected: bool = True


class DataFactory:
    def __init__(self, config: DataFactoryConfig | None = None) -> None:
        self.config = config or DataFactoryConfig()
        self.episodes_dir = Path(self.config.episodes_dir)
        self.station_setting_path = Path(self.config.station_setting_path)
        self._settings = self._load_settings(self.station_setting_path)
        self._rng = random.Random(self.config.seed)
        self._episode_paths = self._load_episode_paths()
        self._order = list(range(len(self._episode_paths)))
        self._cursor = 0
        self._station_specs = self._build_station_specs()
        self._network = self._build_network()
        if self._shuffle_episode_order:
            self._rng.shuffle(self._order)

    def __call__(self) -> EpisodeData:
        path = self._next_episode_path()
        return EpisodeData(
            station_specs=tuple(self._station_specs),
            vehicle_requests=tuple(self._load_vehicle_requests(path)),
            network=self._network,
            timestep_minutes=float(self._environment_settings().get("timestep_minutes", 1.0)),
        )

    def env_config(self) -> SplitChargingEnvConfig:
        environment = self._environment_settings()
        return SplitChargingEnvConfig(
            max_station_count=int(environment.get("max_station_count", len(self._station_specs))),
            episode_horizon_minutes=float(environment.get("episode_horizon_minutes", 1440.0)),
            max_power_kw=float(environment.get("max_power_kw", 10000.0)),
            max_ess_kwh=float(environment.get("max_ess_kwh", 10000.0)),
        )

    def _next_episode_path(self) -> Path:
        if self._cursor >= len(self._order):
            self._cursor = 0
            if self._shuffle_episode_order:
                self._rng.shuffle(self._order)
        index = self._order[self._cursor]
        self._cursor += 1
        return self._episode_paths[index]

    def _build_station_specs(self) -> list[StationSpec]:
        station_ids = [int(item) for item in self._settings["station_ids"]]
        stations = self._settings["stations"]
        renewable = self._settings.get("renewable") or {}
        ess = self._settings.get("ess", {})
        if "pv_indicator" not in renewable:
            if renewable.get("pv_power_csv"):
                raise ValueError("pv_indicator is required when pv_power_csv is configured")
            raw_pv_indicator = [0] * len(station_ids)
        else:
            raw_pv_indicator = renewable["pv_indicator"]
        pv_trace = self._load_pv_trace(renewable)
        if not isinstance(raw_pv_indicator, Sequence) or isinstance(raw_pv_indicator, (str, bytes)):
            raise ValueError("pv_indicator must be a sequence of 0/1 values")
        if len(raw_pv_indicator) != len(station_ids):
            raise ValueError("pv_indicator length must equal station_ids length")
        if any(type(item) is not int or item not in (0, 1) for item in raw_pv_indicator):
            raise ValueError("pv_indicator values must be integers 0 or 1")
        pv_indicator = tuple(int(item) for item in raw_pv_indicator)
        ess_indicator = [int(item) for item in ess.get("ess_indicator", [0] * len(station_ids))]

        specs: list[StationSpec] = []
        for index, station_id in enumerate(station_ids):
            has_ess = bool(ess_indicator[index]) if index < len(ess_indicator) else False
            specs.append(
                StationSpec(
                    station_id=int(station_id),
                    charge_capacity=int(_list_value(stations["charge_capacity"], index)),
                    p_plug_kw=float(stations.get("p_plug_kw", 120.0)),
                    p_max_kw=float(_list_value(stations["p_max_kw"], index)),
                    p_grid_max_kw=float(_list_value(stations["p_grid_max_kw"], index)),
                    eta=float(stations.get("eta", 0.95)),
                    renewable_power_trace=pv_trace if pv_indicator[index] == 1 else None,
                    ess_capacity_kwh=float(ess.get("capacity_kwh", 0.0)) if has_ess else 0.0,
                    ess_initial_kwh=float(ess.get("initial_kwh", 0.0)) if has_ess else 0.0,
                    p_ess_charge_max_kw=float(ess.get("p_charge_max_kw", 0.0)) if has_ess else 0.0,
                    p_ess_discharge_max_kw=float(ess.get("p_discharge_max_kw", 0.0)) if has_ess else 0.0,
                    ess_charge_efficiency=float(ess.get("charge_efficiency", 0.95)),
                    ess_discharge_efficiency=float(ess.get("discharge_efficiency", 0.95)),
                    ess_power_trace=None,
                )
            )
        return specs

    def _build_network(self) -> Any:
        station_ids = [spec.station_id for spec in self._station_specs]
        if bool(self.config.use_vdf_road_map):
            if self.config.road_map_nodes_path is None:
                raise ValueError("road_map_nodes_path is required when use_vdf_road_map=True")
            if self.config.road_map_links_path is None:
                raise ValueError("road_map_links_path is required when use_vdf_road_map=True")
            if self.config.link_flows_path is None:
                raise ValueError("link_flows_path is required when use_vdf_road_map=True")
            return RoadMapVDFNetwork(
                station_ids=station_ids,
                nodes_path=_resolve_config_path(self.config.road_map_nodes_path, anchor=self.station_setting_path),
                links_path=_resolve_config_path(self.config.road_map_links_path, anchor=self.station_setting_path),
                link_flows_path=_resolve_config_path(self.config.link_flows_path, anchor=self.station_setting_path),
                link_parameters_path=(
                    None
                    if self.config.link_parameters_path is None
                    else _resolve_config_path(self.config.link_parameters_path, anchor=self.station_setting_path)
                ),
                undirected=bool(self.config.road_map_undirected),
            )
        return BidirectionalHighwayNetwork(
            station_ids=station_ids,
            travel_time_model_path=self.config.travel_time_model_path,
        )

    def _load_pv_trace(self, renewable: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
        csv_value = renewable.get("pv_power_csv")
        if not csv_value:
            return ()
        path = _resolve_path(csv_value, anchor=self.station_setting_path)
        time_column = str(renewable.get("time_column", "time"))
        power_column = str(renewable.get("pv_power_column", "pv_power_kw"))
        rows: list[tuple[float, float]] = []
        with path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                rows.append(
                    (
                        _parse_minutes(row[time_column]),
                        max(0.0, float(row[power_column])),
                    )
                )
        return tuple(rows)

    def _load_vehicle_requests(self, episode_path: Path) -> list[VehicleRequest]:
        requests: list[VehicleRequest] = []
        with episode_path.open(newline="", encoding="utf-8") as csv_file:
            for row_number, row in enumerate(csv.DictReader(csv_file), start=1):
                origin = _parse_station_id(row["o_i"])
                destination = _parse_station_id(row["d_i"])
                decision_time = _parse_minutes(row["arrival_time"])
                battery_capacity = float(row["B_i"])
                initial_soc = float(row["start_soc"])
                target_soc = float(row["target_soc"])
                path_nodes = _parse_node_sequence(row.get("path_node_ids", ""))
                if not path_nodes:
                    required_stations = _parse_node_sequence(row.get("required_station_ids", ""))
                    if required_stations and hasattr(self._network, "path_nodes_via"):
                        path_nodes = self._network.path_nodes_via(
                            origin,
                            destination,
                            required_stations,
                            decision_time,
                        )
                    else:
                        path_nodes = self._network.path_nodes(origin, destination)
                candidate_stations = _candidate_stations(path_nodes)
                spec = VehicleSpec(
                    battery_capacity=float(battery_capacity),
                    initial_soc=float(initial_soc),
                    soc_min=float(self.config.vehicle_soc_min),
                    p_max_kw=float(self.config.vehicle_p_max_kw),
                    p_min_kw=float(self.config.vehicle_p_min_kw),
                    rho_kwh_per_km=float(row["rho_i"]),
                    origin=int(origin),
                    destination=int(destination),
                    departure_time=float(decision_time),
                    path_nodes=tuple(path_nodes),
                    path_edges=_path_edges(path_nodes),
                    candidate_stations=tuple(candidate_stations),
                    demand_kwh=max(0.0, float(battery_capacity) * (target_soc - initial_soc)),
                )
                requests.append(
                    VehicleRequest(
                        vehicle_id=_parse_vehicle_id(row["vehicle_id"], fallback=row_number),
                        decision_time=float(decision_time),
                        vehicle_spec=spec,
                        target_soc=float(target_soc),
                    )
                )
        return sorted(
            requests,
            key=lambda item: (float(item.decision_time), int(item.vehicle_id)),
        )

    def _environment_settings(self) -> Mapping[str, Any]:
        return self._settings.get("environment", {})

    def _load_episode_paths(self) -> list[Path]:
        if self.config.request_manifest_path is None:
            self._shuffle_episode_order = bool(self.config.shuffle)
            return self._discover_episode_paths(self.episodes_dir)
        split = self.config.request_split
        if split not in ("train", "validation", "test"):
            raise ValueError("request_split must be train, validation, or test")
        manifest = RequestManifest.load(self.config.request_manifest_path, self.episodes_dir)
        self._shuffle_episode_order = bool(self.config.shuffle) and split == "train"
        return list(getattr(manifest, split))

    @staticmethod
    def _load_settings(path: Path) -> Mapping[str, Any]:
        with Path(path).open(encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file)
        if not isinstance(data, Mapping):
            raise ValueError("station_setting_path must contain a YAML mapping.")
        return data

    @staticmethod
    def _discover_episode_paths(directory: Path) -> list[Path]:
        paths = sorted(Path(directory).glob("episode_*.csv"))
        if not paths:
            paths = sorted(Path(directory).glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No episode CSV files found under {directory}.")
        return paths


class BidirectionalHighwayNetwork:
    def __init__(
        self,
        *,
        station_ids: Sequence[int],
        scenario: HighwayTravelTimeScenario | None = None,
        travel_time_model_path: str | Path | None = None,
    ) -> None:
        self.station_ids = tuple(int(item) for item in station_ids)
        if len(self.station_ids) < 1:
            raise ValueError("station_ids must not be empty.")
        self._scenario = scenario or build_default_highway_scenario()
        if len(self.station_ids) - 1 > len(self._scenario.segments):
            raise ValueError("station_ids require more highway segments than the scenario provides.")
        self._segments = tuple(self._scenario.segments[: max(0, len(self.station_ids) - 1)])
        self._edge_to_segment_index = {
            tuple(sorted((self.station_ids[index], self.station_ids[index + 1]))): index
            for index in range(len(self.station_ids) - 1)
        }
        self._travel_time_model = (
            None
            if travel_time_model_path is None
            else _TravelTimeModelEstimator(Path(travel_time_model_path), self._scenario)
        )

    def path_nodes(self, u: int, v: int) -> tuple[int, ...]:
        source = int(u)
        target = int(v)
        if source not in self.station_ids or target not in self.station_ids:
            raise ValueError("origin and destination must be known station ids.")
        start = self.station_ids.index(source)
        end = self.station_ids.index(target)
        step = 1 if end >= start else -1
        return tuple(self.station_ids[index] for index in range(start, end + step, step))

    def path_time(self, u: int, v: int, t: float, route_nodes=None) -> float:
        nodes = tuple(int(item) for item in (route_nodes or self.path_nodes(int(u), int(v))))
        indices = self._route_segment_indices(nodes)
        if not indices:
            return 0.0
        if self._travel_time_model is not None:
            return self._travel_time_model.predict(indices, float(t))
        peak = self._scenario.peak_multiplier(float(t))
        return float(
            sum(
                self._segments[index].free_flow_time_min
                * peak
                * self._segments[index].bottleneck_multiplier
                for index in indices
            )
        )

    def path_energy(self, u: int, v: int, t: float, vehicle_or_rho, route_nodes=None) -> float:
        nodes = tuple(int(item) for item in (route_nodes or self.path_nodes(int(u), int(v))))
        rho = float(getattr(vehicle_or_rho, "rho_kwh_per_km", vehicle_or_rho))
        return float(
            sum(self._segments[index].length_km for index in self._route_segment_indices(nodes))
            * rho
        )

    def _route_segment_indices(self, nodes: tuple[int, ...]) -> tuple[int, ...]:
        if len(nodes) <= 1:
            return ()
        indices: list[int] = []
        for left, right in zip(nodes, nodes[1:]):
            key = tuple(sorted((int(left), int(right))))
            try:
                indices.append(self._edge_to_segment_index[key])
            except KeyError as exc:
                raise ValueError(f"route contains non-adjacent stations: {left}, {right}") from exc
        return tuple(indices)


class RoadMapVDFNetwork:
    def __init__(
        self,
        *,
        station_ids: Sequence[int],
        nodes_path: str | Path,
        links_path: str | Path,
        link_flows_path: str | Path,
        link_parameters_path: str | Path | None = None,
        undirected: bool = True,
    ) -> None:
        self.station_ids = tuple(int(item) for item in station_ids)
        if len(self.station_ids) < 1:
            raise ValueError("station_ids must not be empty.")
        self._sp = _load_shortest_path_module()
        self._nodes_path = Path(nodes_path)
        self._links_path = Path(links_path)
        self._link_flows_path = Path(link_flows_path)
        self._link_parameters_path = None if link_parameters_path is None else Path(link_parameters_path)
        self._undirected = bool(undirected)
        self._distance_map = self._sp.load_road_map(
            self._nodes_path,
            self._links_path,
            undirected=self._undirected,
        )
        self._road_maps_by_hour: dict[int, Any] = {}

    def path_nodes(self, u: int, v: int) -> tuple[int, ...]:
        result = self._sp.shortest_path_between_nodes(self._distance_map, int(u), int(v))
        return tuple(int(node_id) for node_id in result.node_ids)

    def path_nodes_via(
        self,
        u: int,
        v: int,
        required_station_ids: Sequence[int],
        departure_time_min: float,
    ) -> tuple[int, ...]:
        road_map = self._road_map_for_time(float(departure_time_min))
        result = self._sp.shortest_path_via_charging_stations(
            road_map,
            int(u),
            int(v),
            [int(node_id) for node_id in required_station_ids],
        )
        return tuple(int(node_id) for node_id in result.node_ids)

    def path_time(self, u: int, v: int, t: float, route_nodes=None) -> float:
        road_map = self._road_map_for_time(float(t))
        if route_nodes is None:
            result = self._sp.shortest_path_between_nodes(road_map, int(u), int(v))
            return float(result.travel_time_min or 0.0)
        nodes = self._route_slice(int(u), int(v), tuple(int(item) for item in route_nodes))
        return _sum_edge_values(road_map.edge_weights, nodes, "travel time")

    def path_energy(self, u: int, v: int, t: float, vehicle_or_rho, route_nodes=None) -> float:
        if route_nodes is None:
            nodes = self.path_nodes(int(u), int(v))
        else:
            nodes = self._route_slice(int(u), int(v), tuple(int(item) for item in route_nodes))
        rho = float(getattr(vehicle_or_rho, "rho_kwh_per_km", vehicle_or_rho))
        distance_km = _sum_edge_values(self._distance_map.edge_lengths_m, nodes, "distance") / 1000.0
        return float(distance_km * rho)

    def _road_map_for_time(self, minutes: float) -> Any:
        hour = int((float(minutes) % (24.0 * 60.0)) // 60.0)
        if hour not in self._road_maps_by_hour:
            self._road_maps_by_hour[hour] = self._sp.load_road_map(
                self._nodes_path,
                self._links_path,
                undirected=self._undirected,
                weight_mode="travel_time",
                hour_of_day=hour,
                link_flows_path=self._link_flows_path,
                link_parameters_path=self._link_parameters_path,
            )
        return self._road_maps_by_hour[hour]

    def _route_slice(self, source: int, target: int, route_nodes: tuple[int, ...]) -> tuple[int, ...]:
        if int(source) == int(target):
            return (int(source),)
        for start_index, node_id in enumerate(route_nodes):
            if int(node_id) != int(source):
                continue
            for end_index in range(start_index + 1, len(route_nodes)):
                if int(route_nodes[end_index]) == int(target):
                    return tuple(route_nodes[start_index : end_index + 1])
            for end_index in range(start_index - 1, -1, -1):
                if int(route_nodes[end_index]) == int(target):
                    return tuple(reversed(route_nodes[end_index : start_index + 1]))
        return self.path_nodes(int(source), int(target))


class _TravelTimeModelEstimator:
    def __init__(self, checkpoint_path: Path, scenario: HighwayTravelTimeScenario) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = TravelTimeModelConfig(**checkpoint["model_config"])
        self._scenario = scenario
        self._config = config
        self._model = GraphConvolutionalTransformer(config)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        edge_sources, edge_targets = scenario.edge_index()
        self._edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self._segment_features = _segment_features(scenario)

    def predict(self, route_segment_indices: Sequence[int], departure_time_min: float) -> float:
        if len(route_segment_indices) > int(self._config.max_route_len):
            raise ValueError("route length exceeds travel-time model max_route_len.")
        padding = [0] * (int(self._config.max_route_len) - len(route_segment_indices))
        route_mask = [True] * len(route_segment_indices) + [False] * len(padding)
        batch = TravelTimeBatch(
            segment_features=self._segment_features,
            edge_index=self._edge_index,
            route_segment_ids=torch.tensor([list(route_segment_indices) + padding], dtype=torch.long),
            route_mask=torch.tensor([route_mask], dtype=torch.bool),
            departure_features=_departure_features(
                scenario=self._scenario,
                departure_time_min=float(departure_time_min),
                feature_dim=int(self._config.departure_feature_dim),
            ),
        )
        return float(predict_travel_time(self._model, batch).item())


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


def _departure_features(
    *,
    scenario: HighwayTravelTimeScenario,
    departure_time_min: float,
    feature_dim: int,
) -> torch.Tensor | None:
    if int(feature_dim) == 0:
        return None
    hour_angle = 2.0 * math.pi * ((float(departure_time_min) % 1440.0) / 1440.0)
    values = [
        math.sin(hour_angle),
        math.cos(hour_angle),
        scenario.peak_multiplier(float(departure_time_min)),
        1.0,
        0.0,
    ]
    if int(feature_dim) > len(values):
        values.extend([0.0] * (int(feature_dim) - len(values)))
    return torch.tensor([values[: int(feature_dim)]], dtype=torch.float32)


def _candidate_stations(path_nodes: tuple[int, ...]) -> tuple[int, ...]:
    if len(path_nodes) <= 1:
        return tuple(path_nodes)
    return tuple(path_nodes[:-1])


def _path_edges(path_nodes: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{left}-{right}" for left, right in zip(path_nodes, path_nodes[1:]))


def _parse_station_id(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    return int(text)


def _parse_vehicle_id(value: object, *, fallback: int) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        return int(fallback)
    return int(digits)


def _parse_minutes(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    hour, minute = text.split(":", maxsplit=1)
    return float(int(hour) * 60 + int(minute))


def _parse_node_sequence(value: object) -> tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    normalized = text.replace(";", "-").replace(",", "-")
    return tuple(_parse_station_id(item) for item in normalized.split("-") if item.strip())


def _sum_edge_values(edge_values: Mapping[tuple[int, int], float], nodes: tuple[int, ...], label: str) -> float:
    total = 0.0
    for left, right in zip(nodes, nodes[1:]):
        key = (int(left), int(right))
        try:
            total += float(edge_values[key])
        except KeyError as exc:
            raise ValueError(f"missing {label} for edge {left}->{right}") from exc
    return total


def _list_value(value: Any, index: int) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[int(index)]
    return value


def _resolve_config_path(value: object, *, anchor: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(anchor).parent / path


def _resolve_path(value: object, *, anchor: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(anchor).parent / path


def _load_shortest_path_module() -> Any:
    spec = importlib.util.spec_from_file_location("road_map_shortest_path", SHORT_PATH_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


__all__ = [
    "BidirectionalHighwayNetwork",
    "DataFactory",
    "DataFactoryConfig",
    "RoadMapVDFNetwork",
]
