from __future__ import annotations

from typing import Any

from data.network import RoadNetwork
from chargingpilot.simulator.models import VehicleSpec


_TOL = 1e-9


def reachable_first_stations(
    vehicle: Any,
    network: RoadNetwork,
    t_dep: float,
) -> tuple[int, ...]:
    spec = _vehicle_spec(vehicle)
    path_nodes = _path_nodes(spec)
    reachable: list[int] = []
    seen: set[int] = set()

    for station_id in _station_nodes_on_path(path_nodes, network):
        if station_id in seen:
            continue
        seen.add(station_id)
        required_soc = float(spec.soc_min) + (
            _path_energy(network, spec, int(spec.origin), station_id, float(t_dep))
            / float(spec.battery_capacity)
        )
        if float(spec.initial_soc) + _TOL >= required_soc:
            reachable.append(station_id)

    return tuple(reachable)


def reachable_second_stations(
    vehicle: Any,
    s1: int,
    z1_dep: float,
    t_dep_1: float,
    network: RoadNetwork,
) -> tuple[int, ...]:
    spec = _vehicle_spec(vehicle)
    downstream_nodes = _downstream_path_nodes(spec, int(s1))
    reachable: list[int] = []
    seen: set[int] = set()

    for station_id in _station_nodes_on_path(downstream_nodes, network):
        if station_id in seen:
            continue
        seen.add(station_id)
        required_soc = float(spec.soc_min) + (
            _path_energy(network, spec, int(s1), station_id, float(t_dep_1))
            / float(spec.battery_capacity)
        )
        if float(z1_dep) + _TOL >= required_soc:
            reachable.append(station_id)

    return tuple(reachable)


def min_first_charge_for_split(
    vehicle: Any,
    s1: int,
    s2: int,
    z_arr_1: float,
    t_dep_1: float,
    network: RoadNetwork,
) -> float:
    spec = _vehicle_spec(vehicle)
    required_energy = _path_energy(network, spec, int(s1), int(s2), float(t_dep_1))
    lower_bound = required_energy + (
        float(spec.battery_capacity) * float(spec.soc_min)
    ) - (float(spec.battery_capacity) * float(z_arr_1))
    return max(0.0, float(lower_bound))


def min_first_charge_for_no_split(
    vehicle: Any,
    s1: int,
    d: int,
    z_arr_1: float,
    t_dep_1: float,
    network: RoadNetwork,
) -> float:
    spec = _vehicle_spec(vehicle)
    required_energy = _path_energy(network, spec, int(s1), int(d), float(t_dep_1))
    lower_bound = required_energy + (
        float(spec.battery_capacity) * float(spec.soc_min)
    ) - (float(spec.battery_capacity) * float(z_arr_1))
    return max(0.0, float(lower_bound))


def _vehicle_spec(vehicle: Any) -> VehicleSpec:
    if isinstance(vehicle, VehicleSpec):
        return vehicle
    spec = getattr(vehicle, "spec", None)
    if isinstance(spec, VehicleSpec):
        return spec
    raise ValueError("vehicle must be a VehicleSpec or expose a non-null VehicleSpec as spec.")


def _path_nodes(spec: VehicleSpec) -> tuple[int, ...]:
    path_nodes = tuple(int(node_id) for node_id in spec.path_nodes)
    if not path_nodes:
        raise ValueError("vehicle path_nodes must not be empty.")
    return path_nodes


def _station_nodes_on_path(
    path_nodes: tuple[int, ...],
    network: RoadNetwork,
) -> tuple[int, ...]:
    station_nodes: list[int] = []
    for node_id in path_nodes:
        node = network.nodes[int(node_id)]
        if bool(node.is_station):
            station_nodes.append(int(node_id))
    return tuple(station_nodes)


def _downstream_path_nodes(spec: VehicleSpec, origin: int) -> tuple[int, ...]:
    path_nodes = _path_nodes(spec)
    try:
        origin_index = path_nodes.index(int(origin))
    except ValueError as exc:
        raise ValueError(f"Route does not contain origin node {origin}.") from exc
    return tuple(path_nodes[origin_index + 1 :])


def _path_energy(
    network: RoadNetwork,
    spec: VehicleSpec,
    u: int,
    v: int,
    t: float,
) -> float:
    return float(
        network.path_energy(
            int(u),
            int(v),
            float(t),
            vehicle_or_rho=spec,
            route_nodes=_path_nodes(spec),
        )
    )
