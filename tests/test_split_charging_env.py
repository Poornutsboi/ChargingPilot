import unittest

import numpy as np

from chargingpilot.environment.models import EpisodeData, PendingDecision, RewardWeights, SplitChargingEnvConfig, VehicleRequest
from chargingpilot.environment.split_charging_env import SOC_BINS, SplitChargingRequestEnv
from chargingpilot.simulator.models import ChargingAssignment, StationSpec, VehicleSpec, VehicleStatus


class LinearNetwork:
    def __init__(self, distances: dict[tuple[int, int], float]) -> None:
        self.distances = distances

    def path_time(self, u: int, v: int, t: float, route_nodes=None) -> float:
        return self.distances.get((int(u), int(v)), 0.0)

    def path_energy(self, u: int, v: int, t: float, vehicle_or_rho, route_nodes=None) -> float:
        rho = getattr(vehicle_or_rho, "rho_kwh_per_km", vehicle_or_rho)
        return self.distances.get((int(u), int(v)), 0.0) * float(rho)


def make_spec(*, initial_soc: float = 0.5) -> VehicleSpec:
    return VehicleSpec(
        battery_capacity=100.0,
        initial_soc=initial_soc,
        soc_min=0.0,
        p_max_kw=100.0,
        p_min_kw=20.0,
        rho_kwh_per_km=1.0,
        origin=0,
        destination=3,
        departure_time=0.0,
        path_nodes=(0, 1, 2, 3),
        path_edges=("0->1", "1->2", "2->3"),
        candidate_stations=(1, 2),
    )


def make_env(requests: list[VehicleRequest] | None = None) -> SplitChargingRequestEnv:
    station_specs = [
        StationSpec(station_id=1, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
        StationSpec(station_id=2, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
    ]
    network = LinearNetwork(
        {
            (0, 1): 10.0,
            (0, 2): 25.0,
            (1, 2): 10.0,
            (1, 3): 35.0,
            (2, 3): 10.0,
        }
    )
    episode = EpisodeData(
        station_specs=tuple(station_specs),
        vehicle_requests=tuple(
            requests
            or [
                VehicleRequest(vehicle_id=1, decision_time=0.0, vehicle_spec=make_spec(), target_soc=0.9),
            ]
        ),
        network=network,
        timestep_minutes=1.0,
    )
    return SplitChargingRequestEnv(
        episode_factory=lambda: episode,
        config=SplitChargingEnvConfig(max_station_count=2),
    )


class SplitChargingEnvTests(unittest.TestCase):
    def test_default_reward_weights_prioritize_wait_and_curtailment(self) -> None:
        weights = RewardWeights()

        self.assertAlmostEqual(weights.wait_time, 5.0)
        self.assertAlmostEqual(weights.renewable_curtailment, 5.0)
        self.assertAlmostEqual(weights.grid_energy, 1.0)
        self.assertAlmostEqual(weights.charge_time, 0.5)
        self.assertAlmostEqual(weights.stop_count, 0.1)
        self.assertAlmostEqual(weights.violation, 1000.0)

    def test_soc_bins_are_thirty_to_one_hundred_percent_by_five(self) -> None:
        self.assertEqual(len(SOC_BINS), 15)
        np.testing.assert_allclose(SOC_BINS[:3], np.array([0.30, 0.35, 0.40]))
        self.assertAlmostEqual(float(SOC_BINS[-1]), 1.0)

    def test_observation_space_uses_pruned_renewable_aware_layout(self) -> None:
        env = make_env()

        obs, _info = env.reset(seed=7)

        self.assertEqual(env.observation_space.shape, (26,))
        self.assertEqual(obs.shape, (26,))
        np.testing.assert_allclose(
            obs[:4],
            np.array([0.5, 0.9, 100.0 / 150.0, 100.0 / 500.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            obs[4:15],
            np.array(
                [
                    1.0,
                    1.0,
                    10.0 / 240.0,
                    10.0 / 100.0,
                    0.0,
                    0.0,
                    1.0,
                    100.0 / 500.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        )

    def test_terminal_observation_matches_pruned_shape(self) -> None:
        env = make_env()

        terminal = env._terminal_observation()

        self.assertEqual(terminal.shape, (26,))
        np.testing.assert_allclose(terminal, np.zeros(26, dtype=np.float32))

    def test_no_split_only_opens_dummy_bin_and_decodes_to_target_soc(self) -> None:
        env = make_env()
        env.reset(seed=3)

        pair_index = env.station_pair_index((1, None))
        mask = env.action_masks()
        open_bins = [
            bin_index
            for bin_index in range(len(SOC_BINS))
            if mask[pair_index * len(SOC_BINS) + bin_index]
        ]

        self.assertEqual(open_bins, [0])
        decision = env.decode_action(pair_index * len(SOC_BINS))
        self.assertEqual(decision.s1, 1)
        self.assertIsNone(decision.s2)
        self.assertAlmostEqual(decision.z1_target, 0.9)

    def test_split_mask_blocks_soc_bins_below_arrival_or_above_target(self) -> None:
        env = make_env()
        env.reset(seed=4)

        pair_index = env.station_pair_index((1, 2))
        mask = env.action_masks()
        open_values = [
            round(float(SOC_BINS[bin_index]), 2)
            for bin_index in range(len(SOC_BINS))
            if mask[pair_index * len(SOC_BINS) + bin_index]
        ]

        self.assertEqual(open_values, [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])

    def test_reward_is_returned_to_original_vehicle_when_vehicles_overlap(self) -> None:
        env = make_env(
            [
                VehicleRequest(vehicle_id=1, decision_time=0.0, vehicle_spec=make_spec(), target_soc=0.55),
                VehicleRequest(vehicle_id=2, decision_time=1.0, vehicle_spec=make_spec(), target_soc=0.55),
            ]
        )
        env.reset(seed=5)

        first_action = env.station_pair_index((1, None)) * len(SOC_BINS)
        _obs, _reward, terminated, _truncated, info = env.step(first_action)
        self.assertFalse(terminated)
        self.assertEqual(info["accepted_decision"]["vehicle_id"], 1)

        _obs, _reward, terminated, _truncated, info = env.step(first_action)
        self.assertTrue(terminated)
        finalized_ids = {item["vehicle_id"] for item in info["finalized_transitions"]}

        self.assertEqual(finalized_ids, {1, 2})
        self.assertTrue(all(item["decision_id"] is not None for item in info["finalized_transitions"]))

    def test_master_transition_skips_requests_that_arrive_before_current_vehicle_finishes(self) -> None:
        env = make_env(
            [
                VehicleRequest(vehicle_id=1, decision_time=0.0, vehicle_spec=make_spec(), target_soc=0.9),
                VehicleRequest(vehicle_id=2, decision_time=1.0, vehicle_spec=make_spec(), target_soc=0.55),
                VehicleRequest(vehicle_id=3, decision_time=60.0, vehicle_spec=make_spec(), target_soc=0.6),
            ]
        )
        obs_for_vehicle_1, _info = env.reset(seed=10)

        first_action = env.station_pair_index((1, None)) * len(SOC_BINS)
        _obs, _reward, terminated, _truncated, info = env.step(first_action)

        self.assertFalse(terminated)
        self.assertEqual(env.current_vehicle_id, 2)
        self.assertEqual(info["finalized_transitions"], [])

        _obs, _reward, terminated, _truncated, info = env.step(first_action)

        self.assertFalse(terminated)
        self.assertEqual(env.current_vehicle_id, 3)
        finalized_by_vehicle = {
            int(item["vehicle_id"]): item
            for item in info["finalized_transitions"]
        }
        self.assertEqual(finalized_by_vehicle[1]["next_vehicle_id"], 3)
        self.assertGreater(finalized_by_vehicle[1]["finished_at"], 1.0)
        self.assertLessEqual(finalized_by_vehicle[1]["finished_at"], 60.0)
        np.testing.assert_allclose(
            finalized_by_vehicle[1]["next_observation"],
            env._build_observation(),
        )
        self.assertFalse(finalized_by_vehicle[1]["done"])
        self.assertNotEqual(
            float(obs_for_vehicle_1[1]),
            float(finalized_by_vehicle[1]["next_observation"][1]),
        )

    def test_master_interval_reward_sums_lazy_rewards_completed_before_next_observation(self) -> None:
        env = make_env(
            [
                VehicleRequest(vehicle_id=1, decision_time=0.0, vehicle_spec=make_spec(), target_soc=0.9),
                VehicleRequest(vehicle_id=2, decision_time=1.0, vehicle_spec=make_spec(), target_soc=0.55),
                VehicleRequest(vehicle_id=3, decision_time=80.0, vehicle_spec=make_spec(), target_soc=0.6),
            ]
        )
        env.reset(seed=11)

        first_action = env.station_pair_index((1, None)) * len(SOC_BINS)
        _obs, _reward, terminated, _truncated, info = env.step(first_action)
        self.assertFalse(terminated)
        self.assertEqual(info["finalized_transitions"], [])

        _obs, _reward, terminated, _truncated, info = env.step(first_action)

        self.assertFalse(terminated)
        finalized_by_vehicle = {
            int(item["vehicle_id"]): item
            for item in info["finalized_transitions"]
        }
        self.assertEqual(set(finalized_by_vehicle), {1, 2})
        self.assertEqual(finalized_by_vehicle[1]["next_vehicle_id"], 3)
        self.assertEqual(finalized_by_vehicle[2]["next_vehicle_id"], 3)

        base_events = list(finalized_by_vehicle.values())
        discount = float(env.config.interval_reward_discount)
        time_unit = float(env.config.interval_reward_time_unit_minutes)

        def expected_interval_reward(start_time: float) -> float:
            total = 0.0
            for event in base_events:
                if start_time < float(event["finished_at"]) <= 80.0:
                    exponent = max(0.0, (float(event["finished_at"]) - start_time) / time_unit - 1.0)
                    total += (discount**exponent) * float(event["base_reward"])
            return total

        self.assertNotAlmostEqual(
            finalized_by_vehicle[1]["reward"],
            finalized_by_vehicle[1]["base_reward"],
        )
        self.assertAlmostEqual(finalized_by_vehicle[1]["reward"], expected_interval_reward(0.0))
        self.assertAlmostEqual(finalized_by_vehicle[2]["reward"], expected_interval_reward(1.0))

    def test_split_second_leg_can_arrive_before_next_decision_time(self) -> None:
        env = make_env(
            [
                VehicleRequest(vehicle_id=1, decision_time=0.0, vehicle_spec=make_spec(), target_soc=0.9),
                VehicleRequest(vehicle_id=2, decision_time=30.0, vehicle_spec=make_spec(), target_soc=0.55),
            ]
        )
        env.reset(seed=8)

        pair_index = env.station_pair_index((1, 2))
        bin_index = int(np.where(np.isclose(SOC_BINS, 0.45))[0][0])
        split_action = pair_index * len(SOC_BINS) + bin_index

        _obs, _reward, terminated, _truncated, _info = env.step(split_action)

        self.assertFalse(terminated)
        self.assertEqual(env.current_vehicle_id, 2)

    def test_second_leg_arrival_soc_within_tolerance_does_not_fail_validation(self) -> None:
        station_specs = [
            StationSpec(station_id=1, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
            StationSpec(station_id=2, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
        ]
        episode = EpisodeData(
            station_specs=tuple(station_specs),
            vehicle_requests=(
                VehicleRequest(vehicle_id=1, decision_time=0.0, vehicle_spec=make_spec(), target_soc=0.9),
                VehicleRequest(vehicle_id=2, decision_time=100.0, vehicle_spec=make_spec(), target_soc=0.55),
            ),
            network=LinearNetwork(
                {
                    (0, 1): 10.0,
                    (1, 2): 45.0,
                    (1, 3): 35.0,
                    (2, 3): 10.0,
                }
            ),
            timestep_minutes=1.0,
        )
        env = SplitChargingRequestEnv(
            episode_factory=lambda: episode,
            config=SplitChargingEnvConfig(max_station_count=2),
        )
        env.reset(seed=9)

        pair_index = env.station_pair_index((1, 2))
        bin_index = int(np.where(np.isclose(SOC_BINS, 0.45))[0][0])
        split_action = pair_index * len(SOC_BINS) + bin_index

        _obs, _reward, terminated, _truncated, _info = env.step(split_action)

        self.assertFalse(terminated)
        self.assertEqual(env.current_vehicle_id, 2)

    def test_reward_components_are_scaled_before_weighting(self) -> None:
        env = make_env()
        env.reset(seed=6)
        pending = PendingDecision(
            decision_id=1,
            vehicle_id=1,
            entry_time=0.0,
            action_id=0,
            s1=1,
            s2=2,
            z1_target=0.6,
            z2_target=0.9,
            stop_count=2,
        )
        pending.assignments.append(
            ChargingAssignment(
                vehicle_id=1,
                station_id=1,
                charger_id=0,
                arrival_time=0.0,
                start_time=120.0,
                end_time=150.0,
                wait_time=120.0,
                status_at_arrival=VehicleStatus.QUEUEING,
                grid_used_kwh=50.0,
                renewable_curtailed_kwh=200.0,
            )
        )

        result = env._finalize_pending(pending)

        self.assertAlmostEqual(result["normalized_wait_time"], 2.0)
        self.assertAlmostEqual(result["normalized_charge_time"], 0.5)
        self.assertAlmostEqual(result["normalized_stop_count"], 1.0)
        self.assertAlmostEqual(result["normalized_grid_energy"], 1.0)
        self.assertAlmostEqual(result["normalized_renewable_curtailed"], 2.0)
        self.assertAlmostEqual(result["reward"], -21.35)


if __name__ == "__main__":
    unittest.main()
