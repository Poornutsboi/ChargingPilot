from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IncomingRecord:
    vehicle_id: int
    leg_index: int
    station_id: int
    expected_arrival_time: float
    expected_kwh: float
    provisional: bool


@dataclass(frozen=True)
class IncomingSummary:
    station_ids: tuple[int, ...]
    counts: np.ndarray
    kwh: np.ndarray
    eta_min: np.ndarray
    eta_mean: np.ndarray

    def __post_init__(self) -> None:
        station_ids = tuple(int(station_id) for station_id in self.station_ids)
        size = len(station_ids)
        object.__setattr__(self, "station_ids", station_ids)
        object.__setattr__(self, "counts", _readonly_float32(self.counts, (size, 3)))
        object.__setattr__(self, "kwh", _readonly_float32(self.kwh, (size, 3)))
        object.__setattr__(self, "eta_min", _readonly_float32(self.eta_min, (size,)))
        object.__setattr__(
            self, "eta_mean", _readonly_float32(self.eta_mean, (size,))
        )


class IncomingLoadTracker:
    def __init__(self, station_ids: Sequence[int]) -> None:
        if not isinstance(station_ids, (list, tuple)):
            raise ValueError("station_ids must be a list or tuple of integers")
        normalized = tuple(
            self._validate_int("station_id", station_id)
            for station_id in station_ids
        )
        if not normalized or any(
            left >= right for left, right in zip(normalized, normalized[1:])
        ):
            raise ValueError("station_ids must be strictly ascending and unique")
        self._station_ids = normalized
        self._station_index = {
            station_id: index for index, station_id in enumerate(normalized)
        }
        self._records: dict[tuple[int, int], IncomingRecord] = {}
        self._second_station_ids: dict[int, int] = {}

    @property
    def station_ids(self) -> tuple[int, ...]:
        return self._station_ids

    def add_plan(
        self,
        plan: Any,
        first_arrival_time: float,
        first_kwh: float,
        provisional_second_arrival_time: float | None = None,
        second_kwh: float = 0.0,
    ) -> None:
        vehicle_id = self._validate_int("vehicle_id", plan.vehicle_id)
        first_station_id = self._validate_station(plan.s1)
        second_station = getattr(plan, "s2", None)
        second_station_id = (
            None if second_station is None else self._validate_station(second_station)
        )

        candidates = [
            self._validated_record(
                IncomingRecord(
                    vehicle_id=vehicle_id,
                    leg_index=1,
                    station_id=first_station_id,
                    expected_arrival_time=first_arrival_time,
                    expected_kwh=first_kwh,
                    provisional=False,
                )
            )
        ]
        self._validate_kwh(second_kwh)
        if provisional_second_arrival_time is not None:
            if second_station_id is None:
                raise ValueError(
                    "provisional second arrival requires a second-leg station"
                )
            candidates.append(
                self._validated_record(
                    IncomingRecord(
                        vehicle_id=vehicle_id,
                        leg_index=2,
                        station_id=second_station_id,
                        expected_arrival_time=provisional_second_arrival_time,
                        expected_kwh=second_kwh,
                        provisional=True,
                    )
                )
            )

        keys = [(record.vehicle_id, record.leg_index) for record in candidates]
        duplicate = next((key for key in keys if key in self._records), None)
        if duplicate is not None:
            raise ValueError(f"incoming record {duplicate} already exists")
        if vehicle_id in self._second_station_ids:
            raise ValueError(f"incoming plan for vehicle_id={vehicle_id} already exists")

        self._records.update(zip(keys, candidates))
        if second_station_id is not None:
            self._second_station_ids[vehicle_id] = second_station_id

    def mark_arrived(self, vehicle_id: int, leg_index: int) -> IncomingRecord:
        key = (
            self._validate_int("vehicle_id", vehicle_id),
            self._validate_leg_index(leg_index),
        )
        try:
            record = self._records.pop(key)
        except KeyError as exc:
            raise KeyError(f"No incoming record for key={key}.") from exc
        if key[1] == 2:
            self._second_station_ids.pop(key[0], None)
        return record

    def update_second_leg(
        self,
        vehicle_id: int,
        expected_arrival_time: float,
        expected_kwh: float,
    ) -> None:
        vehicle_key = self._validate_int("vehicle_id", vehicle_id)
        self._validate_time(expected_arrival_time)
        self._validate_kwh(expected_kwh)
        key = (vehicle_key, 2)
        existing = self._records.get(key)
        if existing is not None:
            updated = replace(
                existing,
                expected_arrival_time=expected_arrival_time,
                expected_kwh=expected_kwh,
                provisional=False,
            )
        else:
            try:
                station_id = self._second_station_ids[vehicle_key]
            except KeyError as exc:
                raise KeyError(
                    f"No planned second-leg station for vehicle_id={vehicle_key}."
                ) from exc
            updated = IncomingRecord(
                vehicle_id=vehicle_key,
                leg_index=2,
                station_id=station_id,
                expected_arrival_time=expected_arrival_time,
                expected_kwh=expected_kwh,
                provisional=False,
            )
        self._records[key] = self._validated_record(updated)

    def summarize(
        self,
        now: float,
        windows: Sequence[float] = (15.0, 30.0, 60.0),
    ) -> IncomingSummary:
        now_value = self._validate_time(now)
        boundaries = tuple(float(boundary) for boundary in windows)
        if len(boundaries) != 3 or any(not isfinite(value) for value in boundaries):
            raise ValueError("windows must contain exactly three finite boundaries")
        if boundaries[0] < 0.0 or not (
            boundaries[0] < boundaries[1] < boundaries[2]
        ):
            raise ValueError("windows must be nonnegative and strictly ascending")

        size = len(self._station_ids)
        counts = np.zeros((size, 3), dtype=np.float32)
        kwh = np.zeros((size, 3), dtype=np.float32)
        eta_min = np.zeros(size, dtype=np.float32)
        eta_sum = np.zeros(size, dtype=np.float64)
        eta_count = np.zeros(size, dtype=np.int64)

        for record in self._records.values():
            eta = float(record.expected_arrival_time) - now_value
            if eta < 0.0:
                raise ValueError(
                    f"incoming record {(record.vehicle_id, record.leg_index)} is past due"
                )
            station_index = self._station_index[record.station_id]
            if eta_count[station_index] == 0 or eta < eta_min[station_index]:
                eta_min[station_index] = np.float32(eta)
            eta_sum[station_index] += eta
            eta_count[station_index] += 1

            bucket: int | None
            if eta < boundaries[0]:
                bucket = 0
            elif eta < boundaries[1]:
                bucket = 1
            elif eta <= boundaries[2]:
                bucket = 2
            else:
                bucket = None
            if bucket is not None:
                counts[station_index, bucket] += np.float32(1.0)
                kwh[station_index, bucket] += np.float32(record.expected_kwh)

        eta_mean = np.zeros(size, dtype=np.float32)
        present = eta_count > 0
        eta_mean[present] = (eta_sum[present] / eta_count[present]).astype(np.float32)
        return IncomingSummary(self._station_ids, counts, kwh, eta_min, eta_mean)

    def snapshot(self) -> dict[str, Any]:
        return {
            "station_ids": list(self._station_ids),
            "records": [
                asdict(self._records[key]) for key in sorted(self._records)
            ],
            "second_station_ids": [
                {"vehicle_id": vehicle_id, "station_id": station_id}
                for vehicle_id, station_id in sorted(self._second_station_ids.items())
            ],
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> IncomingLoadTracker:
        if type(snapshot) is not dict:
            raise ValueError("snapshot must be a plain dictionary")
        expected_keys = {"station_ids", "records", "second_station_ids"}
        if set(snapshot) != expected_keys:
            raise ValueError(f"snapshot keys must be {sorted(expected_keys)}")
        if type(snapshot["station_ids"]) is not list:
            raise ValueError("snapshot station_ids must be a list")
        if type(snapshot["records"]) is not list:
            raise ValueError("snapshot records must be a list")
        if type(snapshot["second_station_ids"]) is not list:
            raise ValueError("snapshot second_station_ids must be a list")
        tracker = cls(snapshot["station_ids"])

        second_station_ids: dict[int, int] = {}
        for item in snapshot["second_station_ids"]:
            if type(item) is not dict or set(item) != {
                "vehicle_id",
                "station_id",
            }:
                raise ValueError("invalid second_station_ids entry")
            vehicle_id = tracker._validate_int("vehicle_id", item["vehicle_id"])
            station_id = tracker._validate_station(item["station_id"])
            if vehicle_id in second_station_ids:
                raise ValueError(f"duplicate second-leg vehicle_id={vehicle_id}")
            second_station_ids[vehicle_id] = station_id

        for item in snapshot["records"]:
            if type(item) is not dict or set(item) != set(
                IncomingRecord.__dataclass_fields__
            ):
                raise ValueError("invalid incoming record entry")
            record = tracker._validated_record(IncomingRecord(**item))
            key = (record.vehicle_id, record.leg_index)
            if key in tracker._records:
                raise ValueError(f"incoming record {key} already exists")
            if record.leg_index == 2:
                planned_station = second_station_ids.get(record.vehicle_id)
                if planned_station != record.station_id:
                    raise ValueError("second-leg record does not match its planned station")
            tracker._records[key] = record

        tracker._second_station_ids = second_station_ids
        return tracker

    def _validated_record(self, record: IncomingRecord) -> IncomingRecord:
        vehicle_id = self._validate_int("vehicle_id", record.vehicle_id)
        leg_index = self._validate_leg_index(record.leg_index)
        self._validate_station(record.station_id)
        arrival_time = self._validate_time(record.expected_arrival_time)
        expected_kwh = self._validate_kwh(record.expected_kwh)
        if type(record.provisional) is not bool:
            raise ValueError("provisional must be a bool")
        if leg_index == 1 and record.provisional:
            raise ValueError("first-leg incoming records cannot be provisional")
        return IncomingRecord(
            vehicle_id=vehicle_id,
            leg_index=leg_index,
            station_id=record.station_id,
            expected_arrival_time=arrival_time,
            expected_kwh=expected_kwh,
            provisional=record.provisional,
        )

    def _validate_station(self, station_id: int) -> int:
        normalized = self._validate_int("station_id", station_id)
        if normalized not in self._station_index:
            raise ValueError(f"station_id={normalized} is not tracked")
        return normalized

    @staticmethod
    def _validate_leg_index(leg_index: int) -> int:
        normalized = IncomingLoadTracker._validate_int("leg_index", leg_index)
        if normalized not in (1, 2):
            raise ValueError("leg_index must be 1 or 2")
        return normalized

    @staticmethod
    def _validate_time(value: float) -> float:
        normalized = IncomingLoadTracker._validate_float(
            "expected arrival time", value
        )
        if not isfinite(normalized):
            raise ValueError("expected arrival time must be finite")
        return normalized

    @staticmethod
    def _validate_kwh(value: float) -> float:
        normalized = IncomingLoadTracker._validate_float("expected_kwh", value)
        if not isfinite(normalized) or normalized < 0.0:
            raise ValueError("expected_kwh must be finite and nonnegative")
        return normalized

    @staticmethod
    def _validate_int(name: str, value: Any) -> int:
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        return value

    @staticmethod
    def _validate_float(name: str, value: Any) -> float:
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a real number")
        return float(value)


def _readonly_float32(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"array must have shape {shape}")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(shape)
