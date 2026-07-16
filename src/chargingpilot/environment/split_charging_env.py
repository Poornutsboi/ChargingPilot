from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chargingpilot.environment.models import (
    DecodedAction,
    EpisodeData,
    EpisodeFactory,
    PendingDecision,
    SplitChargingEnvConfig,
    VehicleRequest,
    as_float32,
)
from chargingpilot.simulator.models import ChargingAssignment, ChargingSocRequest, VehicleSpec
from chargingpilot.simulator.simulator import SimulatorCore


SOC_BINS = np.round(np.arange(0.30, 1.0001, 0.05), 2).astype(np.float32)
_STATION_FEATURES = 11
_VEHICLE_FEATURES = 4
_EPS = 1e-6


class SplitChargingRequestEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        episode_factory: EpisodeFactory,
        config: SplitChargingEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.episode_factory = episode_factory
        self.config = config or SplitChargingEnvConfig()
        self._initial_episode = self.episode_factory()
        self._fixed_station_ids = self._station_ids_from_episode(self._initial_episode)
        self._station_slots = int(self.config.max_station_count or len(self._fixed_station_ids))
        if self._station_slots < len(self._fixed_station_ids):
            raise ValueError("max_station_count must be >= the number of station specs.")

        self._station_pairs = self._build_station_pairs(self._fixed_station_ids)
        self.action_space = spaces.Discrete(len(self._station_pairs) * len(SOC_BINS))
        obs_dim = _VEHICLE_FEATURES + self._station_slots * _STATION_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.episode: EpisodeData | None = None
        self.simulator: SimulatorCore | None = None
        self._requests: list[VehicleRequest] = []
        self._request_index = 0
        self._current_request: VehicleRequest | None = None
        self._pending_decisions: dict[int, PendingDecision] = {}
        self._awaiting_first_leg: set[int] = set()
        self._awaiting_second_leg: set[int] = set()
        self._reward_events: list[dict] = []
        self._next_decision_id = 1
        self._used_initial_episode = False

    @property
    def current_vehicle_id(self) -> int | None:
        if self._current_request is None:
            return None
        return int(self._current_request.vehicle_id)

    @property
    def current_time(self) -> float:
        if self.simulator is None:
            return 0.0
        return float(self.simulator.clock)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.episode = self._consume_episode()
        station_ids = self._station_ids_from_episode(self.episode)
        if station_ids != self._fixed_station_ids:
            raise ValueError("All episodes must use the same station id set.")
        self.simulator = SimulatorCore(
            station_specs=list(self.episode.station_specs),
            initial_state=self.episode.initial_state,
            timestep_minutes=float(self.episode.timestep_minutes),
        )
        self._requests = sorted(
            list(self.episode.vehicle_requests),
            key=lambda item: (float(item.decision_time), int(item.vehicle_id)),
        )
        self._request_index = 0
        self._current_request = None
        self._pending_decisions = {}
        self._awaiting_first_leg = set()
        self._awaiting_second_leg = set()
        self._reward_events = []
        self._next_decision_id = 1

        finalized: list[dict] = []
        self._move_to_next_request(finalized)
        return self._build_observation(), {"finalized_transitions": finalized}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_request is None:
            return self._terminal_observation(), 0.0, True, False, {
                "finalized_transitions": [],
            }

        action_id = int(action)
        request = self._current_request
        decoded = self.decode_action(action_id)
        decision_id = self._next_decision_id
        self._next_decision_id += 1
        accepted = {
            "decision_id": int(decision_id),
            "vehicle_id": int(request.vehicle_id),
            "action_id": int(action_id),
            "s1": int(decoded.s1),
            "s2": decoded.s2,
            "z1_target": float(decoded.z1_target),
            "z2_target": float(decoded.z2_target),
            "valid": bool(decoded.valid),
        }

        pending = PendingDecision(
            decision_id=int(decision_id),
            vehicle_id=int(request.vehicle_id),
            entry_time=float(request.decision_time),
            action_id=int(action_id),
            s1=int(decoded.s1),
            s2=None if decoded.s2 is None else int(decoded.s2),
            z1_target=float(decoded.z1_target),
            z2_target=float(decoded.z2_target),
            stop_count=1 if decoded.s2 is None else 2,
            violation=0.0 if decoded.valid else 1.0,
        )
        self._pending_decisions[int(request.vehicle_id)] = pending
        finalized: list[dict] = []

        if decoded.valid:
            self._schedule_first_leg(request, pending)
            self._awaiting_first_leg.add(int(request.vehicle_id))
        else:
            finalized.append(self._finalize_pending(pending))

        self._request_index += 1
        terminated = self._move_to_next_request(finalized)
        reward = float(sum(item["reward"] for item in finalized))
        return self._build_observation(), reward, bool(terminated), False, {
            "accepted_decision": accepted,
            "finalized_transitions": finalized,
        }

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(int(self.action_space.n), dtype=bool)
        if self._current_request is None:
            return mask
        for action_id in range(int(self.action_space.n)):
            if self._is_action_valid(action_id):
                mask[action_id] = True
        if not mask.any():
            mask[0] = True
        return mask

    def station_pair_index(self, pair: tuple[int, int | None]) -> int:
        normalized = (int(pair[0]), None if pair[1] is None else int(pair[1]))
        return self._station_pairs.index(normalized)

    def decode_action(self, action_id: int) -> DecodedAction:
        if self._current_request is None:
            raise RuntimeError("No active vehicle request is available.")
        action_id = int(action_id)
        pair_index = action_id // len(SOC_BINS)
        bin_index = action_id % len(SOC_BINS)
        s1, s2 = self._station_pairs[pair_index]
        target_soc = float(self._current_request.target_soc)
        z1_target = target_soc if s2 is None else float(SOC_BINS[bin_index])
        return DecodedAction(
            action_id=action_id,
            pair_index=int(pair_index),
            bin_index=int(bin_index),
            s1=int(s1),
            s2=s2,
            z1_target=float(z1_target),
            z2_target=float(target_soc),
            valid=bool(self._is_action_valid(action_id)),
        )

    def _consume_episode(self) -> EpisodeData:
        if not self._used_initial_episode:
            self._used_initial_episode = True
            return self._initial_episode
        return self.episode_factory()

    def _move_to_next_request(self, finalized: list[dict]) -> bool:
        if self._request_index < len(self._requests):
            request = self._requests[self._request_index]
            finalized.extend(self._advance_to_time(float(request.decision_time)))
            self._current_request = request
            self._attach_transition_targets(finalized)
            return False
        self._current_request = None
        finalized.extend(self._drain_pending())
        self._attach_transition_targets(finalized)
        return True

    def _attach_transition_targets(self, finalized: list[dict]) -> None:
        if not finalized:
            return
        done = self._current_request is None
        next_vehicle_id = self.current_vehicle_id
        next_observation = self._build_observation()
        transition_end_time = self._transition_end_time()
        for item in finalized:
            if "next_observation" in item:
                continue
            start_time = float(item["entry_time"])
            item["next_observation"] = np.asarray(next_observation, dtype=np.float32).copy()
            item["next_vehicle_id"] = None if next_vehicle_id is None else int(next_vehicle_id)
            item["done"] = bool(done)
            item["transition_start_time"] = float(start_time)
            item["transition_end_time"] = float(transition_end_time)
            item["reward"] = self._interval_cumulative_reward(
                start_time=start_time,
                end_time=float(transition_end_time),
                decision_id=int(item["decision_id"]),
            )

    def _transition_end_time(self) -> float:
        if self._current_request is not None:
            return float(self._current_request.decision_time)
        if self.simulator is None:
            return 0.0
        return float(self.simulator.clock)

    def _interval_cumulative_reward(
        self,
        *,
        start_time: float,
        end_time: float,
        decision_id: int,
    ) -> float:
        discount = float(self.config.interval_reward_discount)
        if discount < 0.0 or discount > 1.0:
            raise ValueError("interval_reward_discount must be in [0, 1].")
        time_unit = _positive_scale(
            self.config.interval_reward_time_unit_minutes,
            "interval_reward_time_unit_minutes",
        )
        total = 0.0
        for event in self._reward_events:
            finished_at = float(event["finished_at"])
            same_instant_own_event = (
                int(event["decision_id"]) == int(decision_id)
                and abs(finished_at - float(start_time)) <= _EPS
            )
            if not same_instant_own_event and not (
                float(start_time) + _EPS < finished_at <= float(end_time) + _EPS
            ):
                continue
            exponent = max(0.0, (finished_at - float(start_time)) / time_unit - 1.0)
            total += (discount**exponent) * float(event["base_reward"])
        return float(total)

    def _advance_to_time(self, target_time: float) -> list[dict]:
        self._require_runtime()
        finalized: list[dict] = []
        target = float(target_time)
        while True:
            clock = float(self.simulator.clock)
            next_due = self.simulator.next_scheduled_soc_arrival_time()
            if next_due is not None and float(next_due) <= clock + _EPS:
                for request in self.simulator.pop_due_scheduled_soc_arrivals(clock):
                    self.simulator.enqueue_soc_arrival(request)
                continue

            if clock >= target - _EPS:
                return finalized

            step_target = min(
                target,
                clock + float(self.simulator.timestep_minutes),
            )
            if next_due is not None:
                step_target = min(step_target, float(next_due))

            completed = self.simulator.advance_to(step_target)
            finalized.extend(self._handle_completions(completed))

    def _drain_pending(self) -> list[dict]:
        self._require_runtime()
        finalized: list[dict] = []
        steps = 0
        while self._pending_decisions:
            if steps >= int(self.config.max_drain_steps):
                raise RuntimeError("Exceeded max_drain_steps while waiting for charging completion.")
            next_due = self.simulator.next_scheduled_soc_arrival_time()
            if next_due is None:
                target = float(self.simulator.clock) + float(self.simulator.timestep_minutes)
            else:
                target = max(float(self.simulator.clock), float(next_due))
            finalized.extend(self._advance_to_time(target))
            steps += 1
        return finalized

    def _handle_completions(self, assignments: list[ChargingAssignment]) -> list[dict]:
        finalized: list[dict] = []
        for assignment in assignments:
            vehicle_id = int(assignment.vehicle_id)
            pending = self._pending_decisions.get(vehicle_id)
            if pending is None:
                continue
            pending.assignments.append(assignment)
            if vehicle_id in self._awaiting_first_leg:
                self._awaiting_first_leg.remove(vehicle_id)
                if pending.s2 is None:
                    finalized.append(self._finalize_pending(pending))
                    continue
                if self._schedule_second_leg(pending, assignment):
                    self._awaiting_second_leg.add(vehicle_id)
                else:
                    finalized.append(self._finalize_pending(pending))
                continue
            if vehicle_id in self._awaiting_second_leg:
                self._awaiting_second_leg.remove(vehicle_id)
                finalized.append(self._finalize_pending(pending))
        return finalized

    def _finalize_pending(self, pending: PendingDecision) -> dict:
        self._pending_decisions.pop(int(pending.vehicle_id), None)
        self._awaiting_first_leg.discard(int(pending.vehicle_id))
        self._awaiting_second_leg.discard(int(pending.vehicle_id))
        weights = self.config.reward_weights
        scales = self.config.reward_scales
        wait_time = sum(float(item.wait_time) for item in pending.assignments)
        charge_time = sum(float(item.end_time) - float(item.start_time) for item in pending.assignments)
        grid_energy = sum(float(item.grid_used_kwh) for item in pending.assignments)
        curtailed = sum(float(item.renewable_curtailed_kwh) for item in pending.assignments)
        finished_at = (
            max(float(item.end_time) for item in pending.assignments)
            if pending.assignments
            else float(pending.entry_time)
        )
        normalized_wait_time = wait_time / _positive_scale(scales.wait_time_minutes, "wait_time_minutes")
        normalized_charge_time = charge_time / _positive_scale(scales.charge_time_minutes, "charge_time_minutes")
        normalized_stop_count = float(pending.stop_count) / _positive_scale(scales.stop_count, "stop_count")
        normalized_grid_energy = grid_energy / _positive_scale(scales.grid_energy_kwh, "grid_energy_kwh")
        normalized_curtailed = curtailed / _positive_scale(
            scales.renewable_curtailment_kwh,
            "renewable_curtailment_kwh",
        )
        normalized_violation = float(pending.violation) / _positive_scale(scales.violation, "violation")
        base_reward = -(
            float(weights.wait_time) * normalized_wait_time
            + float(weights.charge_time) * normalized_charge_time
            + float(weights.stop_count) * normalized_stop_count
            + float(weights.grid_energy) * normalized_grid_energy
            + float(weights.renewable_curtailment) * normalized_curtailed
            + float(weights.violation) * normalized_violation
        )
        event = {
            "decision_id": int(pending.decision_id),
            "vehicle_id": int(pending.vehicle_id),
            "base_reward": float(base_reward),
            "reward": float(base_reward),
            "entry_time": float(pending.entry_time),
            "finished_at": float(finished_at),
            "wait_time": float(wait_time),
            "charge_time": float(charge_time),
            "stop_count": int(pending.stop_count),
            "grid_used_kwh": float(grid_energy),
            "renewable_curtailed_kwh": float(curtailed),
            "violation": float(pending.violation),
            "normalized_wait_time": float(normalized_wait_time),
            "normalized_charge_time": float(normalized_charge_time),
            "normalized_stop_count": float(normalized_stop_count),
            "normalized_grid_energy": float(normalized_grid_energy),
            "normalized_renewable_curtailed": float(normalized_curtailed),
            "normalized_violation": float(normalized_violation),
        }
        self._reward_events.append(event)
        return event

    def _schedule_first_leg(self, request: VehicleRequest, pending: PendingDecision) -> None:
        self._require_runtime()
        spec = request.vehicle_spec
        arrival_time = float(request.decision_time) + self._path_time(
            int(spec.origin),
            int(pending.s1),
            float(request.decision_time),
            spec,
        )
        arrival_soc = self._arrival_soc(
            spec=spec,
            source=int(spec.origin),
            target=int(pending.s1),
            departure_soc=float(spec.initial_soc),
            departure_time=float(request.decision_time),
        )
        arrival_soc = self._clamp_arrival_soc_to_min(spec, arrival_soc)
        charging_request = ChargingSocRequest(
            vehicle_id=int(request.vehicle_id),
            station_id=int(pending.s1),
            arrival_time=float(arrival_time),
            vehicle_spec=spec,
            arrival_soc=float(arrival_soc),
            target_soc=float(pending.z1_target),
        )
        self._schedule_or_enqueue(charging_request)

    def _schedule_second_leg(
        self,
        pending: PendingDecision,
        first_assignment: ChargingAssignment,
    ) -> bool:
        if pending.s2 is None:
            return False
        spec = first_assignment.vehicle_spec if hasattr(first_assignment, "vehicle_spec") else None
        if spec is None:
            spec = self._vehicle_spec_for(int(pending.vehicle_id))
        departure_time = float(first_assignment.end_time)
        departure_soc = float(first_assignment.end_soc or pending.z1_target)
        arrival_time = departure_time + self._path_time(
            int(pending.s1),
            int(pending.s2),
            departure_time,
            spec,
        )
        arrival_soc = self._arrival_soc(
            spec=spec,
            source=int(pending.s1),
            target=int(pending.s2),
            departure_soc=departure_soc,
            departure_time=departure_time,
        )
        if float(pending.z2_target) <= arrival_soc + _EPS:
            return False
        arrival_soc = self._clamp_arrival_soc_to_min(spec, arrival_soc)
        charging_request = ChargingSocRequest(
            vehicle_id=int(pending.vehicle_id),
            station_id=int(pending.s2),
            arrival_time=float(arrival_time),
            vehicle_spec=spec,
            arrival_soc=float(arrival_soc),
            target_soc=float(pending.z2_target),
        )
        self._schedule_or_enqueue(charging_request)
        return True

    def _schedule_or_enqueue(self, request: ChargingSocRequest) -> None:
        self._require_runtime()
        if float(request.arrival_time) <= float(self.simulator.clock) + _EPS:
            self.simulator.enqueue_soc_arrival(request)
        else:
            self.simulator.schedule_soc_arrival(request)

    def _is_action_valid(self, action_id: int) -> bool:
        if self._current_request is None:
            return False
        if int(action_id) < 0 or int(action_id) >= int(self.action_space.n):
            return False
        pair_index = int(action_id) // len(SOC_BINS)
        bin_index = int(action_id) % len(SOC_BINS)
        s1, s2 = self._station_pairs[pair_index]
        spec = self._current_request.vehicle_spec
        target_soc = float(self._current_request.target_soc)

        if s1 not in self._candidate_stations(spec):
            return False
        arrival_soc_s1 = self._arrival_soc(
            spec=spec,
            source=int(spec.origin),
            target=int(s1),
            departure_soc=float(spec.initial_soc),
            departure_time=float(self._current_request.decision_time),
        )
        if arrival_soc_s1 + _EPS < float(spec.soc_min):
            return False
        if s2 is None:
            if bin_index != 0:
                return False
            if target_soc <= arrival_soc_s1 + _EPS:
                return False
            return self._can_reach_after_departure(
                spec=spec,
                source=int(s1),
                target=int(spec.destination),
                departure_soc=target_soc,
                departure_time=float(self._current_request.decision_time),
            )

        if s2 not in self._downstream_stations(spec, int(s1)):
            return False
        z1 = float(SOC_BINS[bin_index])
        if z1 <= arrival_soc_s1 + _EPS:
            return False
        if z1 > target_soc + _EPS:
            return False
        if not self._can_reach_after_departure(
            spec=spec,
            source=int(s1),
            target=int(s2),
            departure_soc=z1,
            departure_time=float(self._current_request.decision_time),
        ):
            return False
        return self._can_reach_after_departure(
            spec=spec,
            source=int(s2),
            target=int(spec.destination),
            departure_soc=target_soc,
            departure_time=float(self._current_request.decision_time),
        )

    def _build_observation(self) -> np.ndarray:
        if self._current_request is None:
            return self._terminal_observation()
        self._require_runtime()
        request = self._current_request
        spec = request.vehicle_spec
        now = float(request.decision_time)
        state = self.simulator.get_state(query_time=max(float(self.simulator.clock), now))
        features = [
            float(spec.initial_soc),
            float(request.target_soc),
            float(spec.battery_capacity) / max(_EPS, float(self.config.max_battery_kwh)),
            float(spec.p_max_kw) / max(_EPS, float(self.config.max_power_kw)),
        ]

        stations_state = state["stations"]
        candidate_stations = self._candidate_stations(spec)
        for slot in range(self._station_slots):
            if slot >= len(self._fixed_station_ids):
                features.extend([0.0] * _STATION_FEATURES)
                continue
            station_id = int(self._fixed_station_ids[slot])
            station_state = stations_state[int(station_id)]
            capacity = max(1.0, float(station_state["charge_capacity"]))
            travel_time = self._path_time(int(spec.origin), station_id, now, spec)
            energy = self._path_energy(int(spec.origin), station_id, now, spec)
            reachable_from_origin = self._arrival_soc(
                spec=spec,
                source=int(spec.origin),
                target=station_id,
                departure_soc=float(spec.initial_soc),
                departure_time=now,
            ) >= float(spec.soc_min) - _EPS
            features.extend(
                [
                    1.0 if station_id in candidate_stations else 0.0,
                    1.0 if reachable_from_origin else 0.0,
                    travel_time / max(_EPS, float(self.config.max_travel_time_minutes)),
                    energy / max(_EPS, float(spec.battery_capacity)),
                    len(station_state["queue_waiting_time"]) / capacity,
                    len(station_state["active_vehicle_ids"]) / capacity,
                    sum(1 for available in station_state["available_info"] if available) / capacity,
                    float(station_state["power_available_kw"]) / max(_EPS, float(self.config.max_power_kw)),
                    float(station_state["grid_used_kw"]) / max(_EPS, float(self.config.max_power_kw)),
                    float(station_state["renewable_curtailed_kw"]) / max(_EPS, float(self.config.max_power_kw)),
                    float(station_state["ess_energy_kwh"]) / max(_EPS, float(self.config.max_ess_kwh)),
                ]
            )
        return as_float32(features)

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _can_reach_after_departure(
        self,
        *,
        spec: VehicleSpec,
        source: int,
        target: int,
        departure_soc: float,
        departure_time: float,
    ) -> bool:
        arrival_soc = self._arrival_soc(
            spec=spec,
            source=int(source),
            target=int(target),
            departure_soc=float(departure_soc),
            departure_time=float(departure_time),
        )
        return arrival_soc + _EPS >= float(spec.soc_min)

    def _arrival_soc(
        self,
        *,
        spec: VehicleSpec,
        source: int,
        target: int,
        departure_soc: float,
        departure_time: float,
    ) -> float:
        energy = self._path_energy(int(source), int(target), float(departure_time), spec)
        return float(departure_soc) - (energy / float(spec.battery_capacity))

    @staticmethod
    def _clamp_arrival_soc_to_min(spec: VehicleSpec, arrival_soc: float) -> float:
        soc_min = float(spec.soc_min)
        if float(arrival_soc) < soc_min and float(arrival_soc) + _EPS >= soc_min:
            return soc_min
        return float(arrival_soc)

    def _path_time(self, u: int, v: int, t: float, spec: VehicleSpec) -> float:
        if self.episode is None or self.episode.network is None or int(u) == int(v):
            return 0.0
        return float(
            self.episode.network.path_time(
                int(u),
                int(v),
                float(t),
                route_nodes=spec.path_nodes,
            )
        )

    def _path_energy(self, u: int, v: int, t: float, spec: VehicleSpec) -> float:
        if self.episode is None or self.episode.network is None or int(u) == int(v):
            return 0.0
        return float(
            self.episode.network.path_energy(
                int(u),
                int(v),
                float(t),
                vehicle_or_rho=spec,
                route_nodes=spec.path_nodes,
            )
        )

    def _candidate_stations(self, spec: VehicleSpec) -> tuple[int, ...]:
        candidates = tuple(int(item) for item in spec.candidate_stations)
        if not candidates:
            candidates = tuple(int(node) for node in spec.path_nodes if int(node) in self._fixed_station_ids)
        return tuple(station for station in candidates if station in self._fixed_station_ids)

    def _downstream_stations(self, spec: VehicleSpec, s1: int) -> tuple[int, ...]:
        path_nodes = tuple(int(node) for node in spec.path_nodes)
        if int(s1) not in path_nodes:
            return ()
        index = path_nodes.index(int(s1))
        candidates = set(self._candidate_stations(spec))
        return tuple(node for node in path_nodes[index + 1 :] if node in candidates)

    def _vehicle_spec_for(self, vehicle_id: int) -> VehicleSpec:
        for request in self._requests:
            if int(request.vehicle_id) == int(vehicle_id):
                return request.vehicle_spec
        raise KeyError(f"Unknown vehicle_id={vehicle_id}.")

    def _require_runtime(self) -> None:
        if self.episode is None or self.simulator is None:
            raise RuntimeError("Environment must be reset before use.")

    @staticmethod
    def _station_ids_from_episode(episode: EpisodeData) -> tuple[int, ...]:
        return tuple(sorted(int(spec.station_id) for spec in episode.station_specs))

    @staticmethod
    def _build_station_pairs(station_ids: tuple[int, ...]) -> list[tuple[int, int | None]]:
        pairs: list[tuple[int, int | None]] = []
        for s1 in station_ids:
            pairs.append((int(s1), None))
            for s2 in station_ids:
                if int(s2) != int(s1):
                    pairs.append((int(s1), int(s2)))
        return pairs


def _positive_scale(value: float, label: str) -> float:
    scale = float(value)
    if scale <= 0.0:
        raise ValueError(f"reward scale {label} must be > 0.")
    return scale


from chargingpilot.environment.hierarchical_split_charging_env import HierarchicalSplitChargingRequestEnv
