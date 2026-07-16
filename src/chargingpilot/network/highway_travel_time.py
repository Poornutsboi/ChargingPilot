from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_STATION_CHAIN = (0, 5, 6, 4, 3, 2, 1)

TRAINING_COLUMNS = [
    "sample_id",
    "departure_time_min",
    "hour_of_day",
    "origin_station",
    "destination_station",
    "route_segment_ids",
    "route_length_km",
    "free_flow_time_min",
    "mean_current_speed_kmph",
    "peak_multiplier",
    "weather_multiplier",
    "incident_flag",
    "actual_travel_time_min",
]


@dataclass(frozen=True)
class HighwaySegment:
    segment_id: str
    start_station: int
    end_station: int
    length_km: float
    free_flow_speed_kmph: float
    lane_count: int
    bottleneck_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if float(self.length_km) <= 0.0:
            raise ValueError("length_km must be > 0.")
        if float(self.free_flow_speed_kmph) <= 0.0:
            raise ValueError("free_flow_speed_kmph must be > 0.")
        if int(self.lane_count) <= 0:
            raise ValueError("lane_count must be > 0.")
        if float(self.bottleneck_multiplier) < 1.0:
            raise ValueError("bottleneck_multiplier must be >= 1.")

    @property
    def free_flow_time_min(self) -> float:
        return float(self.length_km) / float(self.free_flow_speed_kmph) * 60.0


@dataclass(frozen=True)
class HighwayTravelTimeScenario:
    station_chain: tuple[int, ...]
    segments: tuple[HighwaySegment, ...]

    def __post_init__(self) -> None:
        if len(self.station_chain) < 2:
            raise ValueError("station_chain must contain at least two stations.")
        if len(self.segments) != len(self.station_chain) - 1:
            raise ValueError("segments must connect every adjacent station pair.")
        for index, segment in enumerate(self.segments):
            expected_start = int(self.station_chain[index])
            expected_end = int(self.station_chain[index + 1])
            if (
                int(segment.start_station) != expected_start
                or int(segment.end_station) != expected_end
            ):
                raise ValueError("segments must follow station_chain order.")

    def route_segments(self, origin_station: int, destination_station: int) -> tuple[HighwaySegment, ...]:
        start_index = self._station_index(origin_station)
        end_index = self._station_index(destination_station)
        if start_index >= end_index:
            raise ValueError("destination_station must be downstream of origin_station.")
        return tuple(self.segments[start_index:end_index])

    def travel_time_minutes(
        self,
        origin_station: int,
        destination_station: int,
        departure_time_min: float,
        *,
        weather_multiplier: float = 1.0,
        incident_multiplier: float = 1.0,
        noise_multiplier: float = 1.0,
    ) -> float:
        route = self.route_segments(int(origin_station), int(destination_station))
        peak_multiplier = self.peak_multiplier(float(departure_time_min))
        total = 0.0
        for segment in route:
            total += (
                segment.free_flow_time_min
                * peak_multiplier
                * float(weather_multiplier)
                * float(incident_multiplier)
                * float(noise_multiplier)
                * float(segment.bottleneck_multiplier)
            )
        return max(0.0, float(total))

    def generate_training_rows(
        self,
        *,
        sample_count: int,
        seed: int | None = None,
        weather_probability: float = 0.18,
        incident_probability: float = 0.04,
        noise_sigma: float = 0.03,
    ) -> list[dict[str, float | int | str]]:
        if int(sample_count) < 0:
            raise ValueError("sample_count must be >= 0.")
        if not 0.0 <= float(weather_probability) <= 1.0:
            raise ValueError("weather_probability must be between 0 and 1.")
        if not 0.0 <= float(incident_probability) <= 1.0:
            raise ValueError("incident_probability must be between 0 and 1.")
        if float(noise_sigma) < 0.0:
            raise ValueError("noise_sigma must be >= 0.")

        rng = random.Random(seed)
        rows: list[dict[str, float | int | str]] = []
        for sample_id in range(1, int(sample_count) + 1):
            origin_index = rng.randrange(0, len(self.station_chain) - 1)
            destination_index = rng.randrange(origin_index + 1, len(self.station_chain))
            origin = int(self.station_chain[origin_index])
            destination = int(self.station_chain[destination_index])
            departure_time = float(rng.randrange(0, 24 * 60))

            weather_multiplier = (
                rng.uniform(1.05, 1.22)
                if rng.random() < float(weather_probability)
                else 1.0
            )
            incident_flag = 1 if rng.random() < float(incident_probability) else 0
            incident_multiplier = rng.uniform(1.30, 1.85) if incident_flag else 1.0
            noise_multiplier = _clamp(
                rng.gauss(1.0, float(noise_sigma)),
                0.85,
                1.20,
            )

            actual_time = self.travel_time_minutes(
                origin,
                destination,
                departure_time,
                weather_multiplier=weather_multiplier,
                incident_multiplier=incident_multiplier,
                noise_multiplier=noise_multiplier,
            )
            route = self.route_segments(origin, destination)
            route_length = sum(float(segment.length_km) for segment in route)
            free_flow_time = sum(float(segment.free_flow_time_min) for segment in route)
            mean_speed = route_length / max(actual_time / 60.0, 1e-9)

            rows.append(
                {
                    "sample_id": int(sample_id),
                    "departure_time_min": round(departure_time, 3),
                    "hour_of_day": round(departure_time / 60.0, 6),
                    "origin_station": int(origin),
                    "destination_station": int(destination),
                    "route_segment_ids": ";".join(segment.segment_id for segment in route),
                    "route_length_km": round(float(route_length), 6),
                    "free_flow_time_min": round(float(free_flow_time), 6),
                    "mean_current_speed_kmph": round(float(mean_speed), 6),
                    "peak_multiplier": round(self.peak_multiplier(departure_time), 6),
                    "weather_multiplier": round(float(weather_multiplier), 6),
                    "incident_flag": int(incident_flag),
                    "actual_travel_time_min": round(float(actual_time), 6),
                }
            )
        return rows

    def edge_index(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        sources = tuple(range(0, len(self.segments) - 1))
        targets = tuple(range(1, len(self.segments)))
        return sources, targets

    def segment_feature_rows(self, departure_time_min: float) -> list[dict[str, float | int | str]]:
        peak = self.peak_multiplier(float(departure_time_min))
        rows: list[dict[str, float | int | str]] = []
        for index, segment in enumerate(self.segments):
            travel_time = segment.free_flow_time_min * peak * segment.bottleneck_multiplier
            current_speed = segment.length_km / max(travel_time / 60.0, 1e-9)
            rows.append(
                {
                    "segment_index": int(index),
                    "segment_id": segment.segment_id,
                    "start_station": int(segment.start_station),
                    "end_station": int(segment.end_station),
                    "length_km": float(segment.length_km),
                    "free_flow_speed_kmph": float(segment.free_flow_speed_kmph),
                    "current_speed_kmph": float(current_speed),
                    "lane_count": int(segment.lane_count),
                    "bottleneck_multiplier": float(segment.bottleneck_multiplier),
                    "peak_multiplier": float(peak),
                }
            )
        return rows

    def peak_multiplier(self, departure_time_min: float) -> float:
        hour = (float(departure_time_min) % (24.0 * 60.0)) / 60.0
        morning = _gaussian(hour, center=8.0, width=1.15)
        evening = _gaussian(hour, center=18.0, width=1.35)
        midday = _gaussian(hour, center=13.0, width=2.5)
        night_relief = _gaussian(hour, center=3.0, width=2.4)
        multiplier = 1.0 + 0.42 * morning + 0.58 * evening + 0.12 * midday - 0.10 * night_relief
        return max(0.85, float(multiplier))

    def _station_index(self, station_id: int) -> int:
        try:
            return self.station_chain.index(int(station_id))
        except ValueError as exc:
            raise ValueError(f"unknown station id: {station_id}") from exc


def build_default_highway_scenario() -> HighwayTravelTimeScenario:
    return HighwayTravelTimeScenario(
        station_chain=DEFAULT_STATION_CHAIN,
        segments=(
            HighwaySegment("0-5", 0, 5, length_km=28.0, free_flow_speed_kmph=105.0, lane_count=3),
            HighwaySegment("5-6", 5, 6, length_km=22.0, free_flow_speed_kmph=100.0, lane_count=3),
            HighwaySegment("6-4", 6, 4, length_km=35.0, free_flow_speed_kmph=110.0, lane_count=2),
            HighwaySegment(
                "4-3",
                4,
                3,
                length_km=18.0,
                free_flow_speed_kmph=90.0,
                lane_count=2,
                bottleneck_multiplier=1.18,
            ),
            HighwaySegment("3-2", 3, 2, length_km=30.0, free_flow_speed_kmph=105.0, lane_count=3),
            HighwaySegment("2-1", 2, 1, length_km=25.0, free_flow_speed_kmph=100.0, lane_count=2),
        ),
    )


def write_training_csv(
    scenario: HighwayTravelTimeScenario,
    output_path: str | Path,
    *,
    sample_count: int,
    seed: int | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = scenario.generate_training_rows(sample_count=int(sample_count), seed=seed)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    scenario = build_default_highway_scenario()
    output_path = write_training_csv(
        scenario,
        args.output,
        sample_count=int(args.samples),
        seed=args.seed,
    )
    print(f"Wrote {int(args.samples)} highway travel-time samples to {output_path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a runnable highway travel-time dataset.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/travel_time/highway_travel_time.csv"),
    )
    return parser.parse_args(argv)


def _gaussian(value: float, *, center: float, width: float) -> float:
    return math.exp(-0.5 * ((float(value) - float(center)) / float(width)) ** 2)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


if __name__ == "__main__":
    main()
