from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_FREE_FLOW_SPEED_KMPH = 100.0
DEFAULT_LANE_COUNT = 3
DEFAULT_CAPACITY_PER_LANE_VEH_PER_HOUR = 1800.0
DEFAULT_M = 2.5
MAX_VOLUME_CAPACITY_RATIO = 1.999


@dataclass(frozen=True)
class LinkVDFParameters:
    link_id: int
    from_id: int
    to_id: int
    length_m: float
    lane_count: int = DEFAULT_LANE_COUNT
    free_flow_speed_kmph: float = DEFAULT_FREE_FLOW_SPEED_KMPH
    capacity_veh_per_hour: float = DEFAULT_LANE_COUNT * DEFAULT_CAPACITY_PER_LANE_VEH_PER_HOUR
    m: float = DEFAULT_M

    def __post_init__(self) -> None:
        if float(self.length_m) <= 0.0:
            raise ValueError("length_m must be positive")
        if int(self.lane_count) <= 0:
            raise ValueError("lane_count must be positive")
        if float(self.free_flow_speed_kmph) <= 0.0:
            raise ValueError("free_flow_speed_kmph must be positive")
        if float(self.capacity_veh_per_hour) <= 0.0:
            raise ValueError("capacity_veh_per_hour must be positive")
        if float(self.m) <= 0.0:
            raise ValueError("m must be positive")


def modified_vdf_travel_time_min(
    *,
    length_m: float,
    free_flow_speed_kmph: float,
    flow_veh_per_hour: float,
    capacity_veh_per_hour: float,
    m: float = DEFAULT_M,
) -> float:
    if float(length_m) <= 0.0:
        raise ValueError("length_m must be positive")
    if float(free_flow_speed_kmph) <= 0.0:
        raise ValueError("free_flow_speed_kmph must be positive")
    if float(capacity_veh_per_hour) <= 0.0:
        raise ValueError("capacity_veh_per_hour must be positive")
    if float(m) <= 0.0:
        raise ValueError("m must be positive")

    free_flow_time_min = (float(length_m) / 1000.0) / float(free_flow_speed_kmph) * 60.0
    ratio = _clamp(
        float(flow_veh_per_hour) / float(capacity_veh_per_hour),
        0.0,
        MAX_VOLUME_CAPACITY_RATIO,
    )
    exponent = 2.0 / float(m)
    if ratio <= 1.0:
        denominator = 1.0 + math.sqrt(max(0.0, 1.0 - ratio**float(m)))
    else:
        denominator = 1.0 - math.sqrt(max(0.0, 1.0 - (2.0 - ratio) ** float(m)))
    return float(free_flow_time_min * (2.0 / max(denominator, 1e-12)) ** exponent)


def link_travel_time_min(
    parameters: LinkVDFParameters,
    hourly_flows: Mapping[int, Mapping[int, float]],
    hour_of_day: int,
) -> float:
    flow = float(hourly_flows.get(int(parameters.link_id), {}).get(_normalize_hour(hour_of_day), 0.0))
    return modified_vdf_travel_time_min(
        length_m=float(parameters.length_m),
        free_flow_speed_kmph=float(parameters.free_flow_speed_kmph),
        flow_veh_per_hour=flow,
        capacity_veh_per_hour=float(parameters.capacity_veh_per_hour),
        m=float(parameters.m),
    )


def load_hourly_flows(path: str | Path) -> dict[int, dict[int, float]]:
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("hourly flow CSV must have a header")
        fieldnames = set(reader.fieldnames)
        if {"link_id", "hour_of_day", "flow_veh_per_hour"}.issubset(fieldnames):
            return _load_long_hourly_flows(reader)
        hour_columns = sorted(name for name in fieldnames if name.startswith("flow_h"))
        if "link_id" not in fieldnames or not hour_columns:
            raise ValueError("hourly flow CSV must be long format or include flow_hXX columns")
        return _load_wide_hourly_flows(reader, hour_columns)


def load_link_parameters(path: str | Path) -> dict[int, LinkVDFParameters]:
    parameters: dict[int, LinkVDFParameters] = {}
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            link_id = int(row["link_id"])
            lane_count = int(float(row.get("lane_count") or DEFAULT_LANE_COUNT))
            capacity_text = row.get("capacity_veh_per_hour")
            if capacity_text in (None, ""):
                capacity = lane_count * DEFAULT_CAPACITY_PER_LANE_VEH_PER_HOUR
            else:
                capacity = float(capacity_text)
            parameters[link_id] = LinkVDFParameters(
                link_id=link_id,
                from_id=int(row["from_id"]),
                to_id=int(row["to_id"]),
                length_m=float(row["length_m"]),
                lane_count=lane_count,
                free_flow_speed_kmph=float(row.get("free_flow_speed_kmph") or DEFAULT_FREE_FLOW_SPEED_KMPH),
                capacity_veh_per_hour=capacity,
                m=float(row.get("m") or DEFAULT_M),
            )
    return parameters


def default_link_parameters(*, link_id: int, from_id: int, to_id: int, length_m: float) -> LinkVDFParameters:
    lane_count = DEFAULT_LANE_COUNT
    return LinkVDFParameters(
        link_id=int(link_id),
        from_id=int(from_id),
        to_id=int(to_id),
        length_m=float(length_m),
        lane_count=lane_count,
        free_flow_speed_kmph=DEFAULT_FREE_FLOW_SPEED_KMPH,
        capacity_veh_per_hour=lane_count * DEFAULT_CAPACITY_PER_LANE_VEH_PER_HOUR,
        m=DEFAULT_M,
    )


def _load_long_hourly_flows(reader: csv.DictReader) -> dict[int, dict[int, float]]:
    flows: dict[int, dict[int, float]] = {}
    for row in reader:
        link_id = int(row["link_id"])
        hour = _normalize_hour(int(float(row["hour_of_day"])))
        flows.setdefault(link_id, {})[hour] = float(row["flow_veh_per_hour"])
    return flows


def _load_wide_hourly_flows(reader: csv.DictReader, hour_columns: list[str]) -> dict[int, dict[int, float]]:
    flows: dict[int, dict[int, float]] = {}
    for row in reader:
        link_id = int(row["link_id"])
        by_hour = flows.setdefault(link_id, {})
        for column in hour_columns:
            text = row.get(column)
            if text in (None, ""):
                continue
            by_hour[_hour_from_column(column)] = float(text)
    return flows


def _hour_from_column(column: str) -> int:
    try:
        return _normalize_hour(int(column.removeprefix("flow_h")))
    except ValueError as exc:
        raise ValueError(f"invalid hourly flow column: {column}") from exc


def _normalize_hour(hour_of_day: int) -> int:
    return int(hour_of_day) % 24


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))
