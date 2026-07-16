from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TRAFFIC_FLOW = np.array(
    [
        275,
        196,
        172,
        239,
        546,
        1486,
        2891,
        2887,
        2179,
        2091,
        2298,
        2601,
        2561,
        2832,
        3206,
        3055,
        3313,
        2633,
        2258,
        1709,
        1394,
        1223,
        693,
        339,
    ],
    dtype=float,
)
P_CHARGE = 0.02

STATION_IDS = [f"S{i}" for i in range(1, 8)]
ORIGIN_ATTRACTION = np.array([1.0, 2.5, 2.5, 1.0, 0.25, 0.25, 1.0], dtype=float)

BATTERY_VALUES = np.array([45, 55, 65, 75, 90, 100, 110], dtype=int)
BATTERY_PROBS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03], dtype=float)

START_SOC_LOW = 0.02
START_SOC_HIGH = 0.30
TARGET_SOC_LOW = 0.75
TARGET_SOC_HIGH = 1.00
TARGET_SOC_BETA_A = 2.5
TARGET_SOC_BETA_B = 1.2
RHO_MEAN = 0.20
RHO_STD = 0.04
RHO_LOW = 0.12
RHO_HIGH = 0.32

DEFAULT_EPISODES = 500
DEFAULT_TOLERANCE = 0.05
DEFAULT_SAME_OD_RATE = 0.005
DEFAULT_OUTPUT = Path("datasets/train")
DEFAULT_MAX_ATTEMPTS = 1000

OUTPUT_COLUMNS = [
    "episode_id",
    "vehicle_id",
    "arrival_time",
    "start_soc",
    "target_soc",
    "B_i",
    "o_i",
    "d_i",
    "rho_i",
]


def expected_vehicle_count() -> int:
    return int(round(float(np.sum(TRAFFIC_FLOW)) * float(P_CHARGE)))


def expected_episode_charge_duration(vehicle_count: int | None = None) -> float:
    count = expected_vehicle_count() if vehicle_count is None else int(vehicle_count)
    expected_battery = float(np.dot(BATTERY_VALUES, BATTERY_PROBS))
    expected_start_soc = (START_SOC_LOW + START_SOC_HIGH) / 2.0
    target_beta_mean = TARGET_SOC_BETA_A / (TARGET_SOC_BETA_A + TARGET_SOC_BETA_B)
    expected_target_soc = TARGET_SOC_LOW + (TARGET_SOC_HIGH - TARGET_SOC_LOW) * target_beta_mean
    return float(count) * expected_battery * (expected_target_soc - expected_start_soc)


def charge_duration_total(df: pd.DataFrame) -> float:
    return float((df["B_i"] * (df["target_soc"] - df["start_soc"])).sum())


def generate_dataset(
    *,
    episodes: int = DEFAULT_EPISODES,
    seed: int | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    same_od_rate: float = DEFAULT_SAME_OD_RATE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    traffic_jitter_sigma: float = 0.25,
    origin_jitter_sigma: float = 0.25,
) -> pd.DataFrame:
    _validate_generation_args(
        episodes=episodes,
        tolerance=tolerance,
        same_od_rate=same_od_rate,
        max_attempts=max_attempts,
    )
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    next_vehicle_id = 1
    baseline_count = expected_vehicle_count()
    baseline_duration = expected_episode_charge_duration(baseline_count)

    for episode_id in range(1, int(episodes) + 1):
        episode = generate_episode(
            episode_id=episode_id,
            rng=rng,
            tolerance=tolerance,
            same_od_rate=same_od_rate,
            max_attempts=max_attempts,
            baseline_vehicle_count=baseline_count,
            baseline_charge_duration=baseline_duration,
            vehicle_id_start=next_vehicle_id,
            traffic_jitter_sigma=traffic_jitter_sigma,
            origin_jitter_sigma=origin_jitter_sigma,
        )
        next_vehicle_id += len(episode)
        frames.append(episode)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]


def generate_episode(
    *,
    episode_id: int,
    rng: np.random.Generator,
    tolerance: float = DEFAULT_TOLERANCE,
    same_od_rate: float = DEFAULT_SAME_OD_RATE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    baseline_vehicle_count: int | None = None,
    baseline_charge_duration: float | None = None,
    vehicle_id_start: int = 1,
    traffic_jitter_sigma: float = 0.25,
    origin_jitter_sigma: float = 0.25,
) -> pd.DataFrame:
    baseline_count = expected_vehicle_count() if baseline_vehicle_count is None else int(baseline_vehicle_count)
    baseline_duration = (
        expected_episode_charge_duration(baseline_count)
        if baseline_charge_duration is None
        else float(baseline_charge_duration)
    )
    count_min = int(np.ceil(float(baseline_count) * (1.0 - float(tolerance))))
    count_max = int(np.floor(float(baseline_count) * (1.0 + float(tolerance))))
    duration_min = float(baseline_duration) * (1.0 - float(tolerance))
    duration_max = float(baseline_duration) * (1.0 + float(tolerance))

    for attempt in range(1, int(max_attempts) + 1):
        vehicle_count = int(rng.integers(count_min, count_max + 1))
        hour_counts = _sample_hour_counts(
            vehicle_count=vehicle_count,
            rng=rng,
            traffic_jitter_sigma=traffic_jitter_sigma,
        )
        origin_prob = _sample_origin_prob(rng=rng, origin_jitter_sigma=origin_jitter_sigma)
        episode = _build_episode_records(
            episode_id=int(episode_id),
            hour_counts=hour_counts,
            origin_prob=origin_prob,
            rng=rng,
            same_od_rate=float(same_od_rate),
            vehicle_id_start=int(vehicle_id_start),
        )
        duration = charge_duration_total(episode)
        if duration_min <= duration <= duration_max:
            return episode

    raise RuntimeError(
        "Unable to generate an episode within tolerance after "
        f"{max_attempts} attempts: duration target "
        f"[{duration_min:.3f}, {duration_max:.3f}]"
    )


def write_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_episode_files(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for stale_file in directory.glob("episode_*.csv"):
        stale_file.unlink()
    old_combined_file = directory / "vehicle_charging_requests.csv"
    if old_combined_file.exists():
        old_combined_file.unlink()

    episode_ids = sorted(int(item) for item in df["episode_id"].unique())
    width = max(4, len(str(max(episode_ids, default=0))))
    paths: list[Path] = []
    for episode_id in episode_ids:
        episode = df[df["episode_id"] == episode_id]
        path = directory / f"episode_{episode_id:0{width}d}.csv"
        episode.to_csv(path, index=False)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    df = generate_dataset(
        episodes=int(args.episodes),
        seed=args.seed,
        tolerance=float(args.tolerance),
        same_od_rate=float(args.same_od_rate),
        max_attempts=int(args.max_attempts),
    )
    output_paths = write_episode_files(df, args.output)
    _print_summary(df, Path(args.output), len(output_paths))


def _validate_generation_args(
    *,
    episodes: int,
    tolerance: float,
    same_od_rate: float,
    max_attempts: int,
) -> None:
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive.")
    if not 0.0 <= float(tolerance) < 1.0:
        raise ValueError("tolerance must be in [0, 1).")
    if not 0.0 <= float(same_od_rate) <= 1.0:
        raise ValueError("same_od_rate must be in [0, 1].")
    if int(max_attempts) <= 0:
        raise ValueError("max_attempts must be positive.")


def _sample_hour_counts(
    *,
    vehicle_count: int,
    rng: np.random.Generator,
    traffic_jitter_sigma: float,
) -> np.ndarray:
    base_prob = TRAFFIC_FLOW / float(np.sum(TRAFFIC_FLOW))
    jitter = rng.lognormal(mean=0.0, sigma=float(traffic_jitter_sigma), size=len(TRAFFIC_FLOW))
    hour_prob = _normalize(base_prob * jitter)
    return rng.multinomial(int(vehicle_count), hour_prob)


def _sample_origin_prob(
    *,
    rng: np.random.Generator,
    origin_jitter_sigma: float,
) -> np.ndarray:
    jitter = rng.lognormal(mean=0.0, sigma=float(origin_jitter_sigma), size=len(ORIGIN_ATTRACTION))
    return _normalize(ORIGIN_ATTRACTION * jitter)


def _build_episode_records(
    *,
    episode_id: int,
    hour_counts: np.ndarray,
    origin_prob: np.ndarray,
    rng: np.random.Generator,
    same_od_rate: float,
    vehicle_id_start: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    vehicle_counter = int(vehicle_id_start)

    for hour, n_charge in enumerate(hour_counts):
        for _ in range(int(n_charge)):
            o_index = int(rng.choice(np.arange(len(STATION_IDS)), p=origin_prob))
            o_i = STATION_IDS[o_index]
            d_i = _sample_destination_station(
                o_index=o_index,
                rng=rng,
                same_od_rate=float(same_od_rate),
            )
            start_soc = _sample_start_soc(rng)
            target_soc = _sample_target_soc(rng)
            b_i = int(rng.choice(BATTERY_VALUES, p=BATTERY_PROBS))
            rho_i = _sample_rho(rng)

            records.append(
                {
                    "episode_id": int(episode_id),
                    "vehicle_id": f"EV{vehicle_counter:08d}",
                    "arrival_time": _sample_arrival_time(hour, rng),
                    "start_soc": round(float(start_soc), 3),
                    "target_soc": round(float(target_soc), 3),
                    "B_i": b_i,
                    "o_i": o_i,
                    "d_i": d_i,
                    "rho_i": round(float(rho_i), 3),
                }
            )
            vehicle_counter += 1

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    return df.sort_values(["episode_id", "arrival_time", "vehicle_id"]).reset_index(drop=True)


def _sample_destination_station(
    *,
    o_index: int,
    rng: np.random.Generator,
    same_od_rate: float,
) -> str:
    if rng.random() < float(same_od_rate):
        return STATION_IDS[int(o_index)]
    possible = [idx for idx in range(len(STATION_IDS)) if idx != int(o_index)]
    d_index = int(rng.choice(possible))
    return STATION_IDS[d_index]


def _sample_start_soc(rng: np.random.Generator) -> float:
    return float(rng.uniform(START_SOC_LOW, START_SOC_HIGH))


def _sample_target_soc(rng: np.random.Generator) -> float:
    x = float(rng.beta(TARGET_SOC_BETA_A, TARGET_SOC_BETA_B))
    return TARGET_SOC_LOW + (TARGET_SOC_HIGH - TARGET_SOC_LOW) * x


def _sample_rho(rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(loc=RHO_MEAN, scale=RHO_STD), RHO_LOW, RHO_HIGH))


def _sample_arrival_time(hour: int, rng: np.random.Generator) -> str:
    minute = int(rng.integers(0, 60))
    return f"{int(hour):02d}:{minute:02d}"


def _normalize(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return np.asarray(weights, dtype=float) / total


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-episode EV charging request data.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for episode CSV files.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--same-od-rate", type=float, default=DEFAULT_SAME_OD_RATE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser.parse_args(argv)


def _print_summary(df: pd.DataFrame, output_dir: Path, file_count: int) -> None:
    grouped = df.groupby("episode_id")
    counts = grouped.size()
    durations = grouped.apply(charge_duration_total, include_groups=False)
    same_od_count = int((df["o_i"] == df["d_i"]).sum())
    same_od_rate = same_od_count / max(1, len(df))

    print(df.head(20))
    print()
    print(f"Episodes: {df['episode_id'].nunique()}")
    print(f"Total vehicles: {len(df)}")
    print(
        "Vehicles per episode: "
        f"min={int(counts.min())}, mean={counts.mean():.2f}, max={int(counts.max())}"
    )
    print(
        "Charge duration per episode: "
        f"min={durations.min():.2f}, mean={durations.mean():.2f}, max={durations.max():.2f}"
    )
    print(f"Same OD vehicles: {same_od_count} ({same_od_rate:.3%})")
    print()
    print("Arrival counts by o_i:")
    print(df["o_i"].value_counts().sort_index())
    print()
    print("Battery capacity distribution:")
    print(df["B_i"].value_counts().sort_index())
    print()
    print(f"CSV files saved to: {output_dir} ({file_count} files)")


if __name__ == "__main__":
    main()
