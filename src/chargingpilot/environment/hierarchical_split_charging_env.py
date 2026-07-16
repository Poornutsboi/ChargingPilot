from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np

from chargingpilot.environment.interval_reward import interval_reward
from chargingpilot.environment.models import EpisodeData, EpisodeFactory, HierarchicalSplitChargingEnvConfig, VehicleRequest
from chargingpilot.environment.structured_observation import StructuredObservation, build_structured_observation
from chargingpilot.routing import ChargingPlan, FeasiblePlanGenerator, HierarchicalAction, InvalidHierarchicalActionError
from chargingpilot.simulator.incoming import IncomingLoadTracker
from chargingpilot.simulator.models import ChargingAssignment, ChargingSocRequest, IntervalMetrics
from chargingpilot.simulator.simulator import ProjectionUnavailableError, SimulatorCore


_EPS = 1e-9
_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_DYNAMIC_ATTRIBUTES = (
    "_used_initial_episode",
    "simulator",
    "incoming",
    "_requests",
    "_request_index",
    "_current_request",
    "_request_context",
    "_pending",
    "_scheduled_leg_by_vehicle",
    "_events",
    "_np_random",
    "_np_random_seed",
)


class NonFiniteObservationError(RuntimeError):
    pass


class NonFiniteRewardError(RuntimeError):
    pass


@dataclass
class _PendingPlan:
    request: VehicleRequest
    plan: ChargingPlan
    current_leg: int = 1


class HierarchicalSplitChargingRequestEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        episode_factory: EpisodeFactory,
        oracle: Any,
        config: HierarchicalSplitChargingEnvConfig | None = None,
        plan_generator: FeasiblePlanGenerator | None = None,
    ) -> None:
        super().__init__()
        self.episode_factory = episode_factory
        self.oracle = oracle
        self.config = config or HierarchicalSplitChargingEnvConfig()
        self.generator = plan_generator or FeasiblePlanGenerator(
            oracle, detour_limit=float(self.config.max_detour_ratio)
        )
        self._initial_episode = self.episode_factory()
        self._used_initial_episode = False
        self._fixed_station_ids = tuple(int(value) for value in oracle.station_ids)
        if len(self._fixed_station_ids) != 72:
            raise ValueError("hierarchical environment requires exactly 72 stations")
        self.episode: EpisodeData | None = None
        self.simulator: SimulatorCore | None = None
        self.incoming = IncomingLoadTracker(self._fixed_station_ids)
        self._requests: list[VehicleRequest] = []
        self._request_index = 0
        self._current_request: VehicleRequest | None = None
        self._request_context: Any | None = None
        self._pending: dict[int, _PendingPlan] = {}
        self._scheduled_leg_by_vehicle: dict[int, int] = {}
        self._events: list[dict[str, Any]] = []

    @property
    def current_vehicle_id(self) -> int | None:
        return None if self._current_request is None else int(self._current_request.vehicle_id)

    @property
    def current_time(self) -> float:
        return 0.0 if self.simulator is None else float(self.simulator.clock)

    @property
    def current_request_context(self) -> Any | None:
        return self._request_context

    def state_dict(self) -> dict[str, Any]:
        dynamic = {
            name: getattr(self, name)
            for name in _CHECKPOINT_DYNAMIC_ATTRIBUTES
            if hasattr(self, name)
        }
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "station_ids": self._fixed_station_ids,
            "initial_episode": _episode_checkpoint_state(self._initial_episode),
            "episode": _episode_checkpoint_state(self.episode),
            "dynamic": copy.deepcopy(dynamic),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("environment checkpoint state must be a mapping")
        expected_keys = {
            "schema_version",
            "station_ids",
            "initial_episode",
            "episode",
            "dynamic",
        }
        if set(state) != expected_keys:
            raise ValueError("environment checkpoint state has an invalid schema")
        if int(state["schema_version"]) != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("environment checkpoint schema version does not match")
        if tuple(int(value) for value in state["station_ids"]) != self._fixed_station_ids:
            raise ValueError("environment checkpoint station ids do not match")
        dynamic = state["dynamic"]
        if not isinstance(dynamic, Mapping):
            raise TypeError("environment checkpoint dynamic state must be a mapping")
        if not set(dynamic).issubset(_CHECKPOINT_DYNAMIC_ATTRIBUTES):
            raise ValueError("environment checkpoint has unknown dynamic attributes")

        static_network = self._checkpoint_network()
        initial_episode = _episode_from_checkpoint(
            state["initial_episode"], static_network
        )
        if initial_episode is None:
            raise ValueError("environment checkpoint must contain an initial episode")
        episode = _episode_from_checkpoint(state["episode"], static_network)
        restored_dynamic = copy.deepcopy(dict(dynamic))
        self._initial_episode = initial_episode
        self.episode = episode
        for name, value in restored_dynamic.items():
            setattr(self, name, value)
        if self._request_context is not None:
            register_context = getattr(self.generator, "register_context", None)
            if callable(register_context):
                self._request_context = register_context(self._request_context)

    def _checkpoint_network(self) -> Any | None:
        factory_network = getattr(self.episode_factory, "_network", None)
        if factory_network is not None:
            return factory_network
        return self._initial_episode.network

    def reset(self, *, seed=None, options=None) -> tuple[StructuredObservation, dict]:
        super().reset(seed=seed)
        self.episode = self._consume_episode()
        station_specs = tuple(sorted(self.episode.station_specs, key=lambda item: item.station_id))
        if tuple(int(item.station_id) for item in station_specs) != self._fixed_station_ids:
            raise ValueError("episode station ids must match the oracle station order")
        self.simulator = SimulatorCore(
            list(station_specs),
            initial_state=self.episode.initial_state,
            timestep_minutes=float(self.episode.timestep_minutes),
            exact_internal_events=True,
        )
        self._requests = sorted(
            self.episode.vehicle_requests,
            key=lambda item: (float(item.decision_time), int(item.vehicle_id)),
        )
        self._request_index = 0
        self._current_request = None
        self._request_context = None
        self._pending = {}
        self._scheduled_leg_by_vehicle = {}
        self.incoming = IncomingLoadTracker(self._fixed_station_ids)
        self._events = []
        self._select_next_request()
        if self._current_request is None:
            return self._terminal_observation(), {"s1_mask": np.zeros(72, dtype=np.bool_)}
        self._advance_to(float(self._current_request.decision_time))
        return self._build_observation(), {"s1_mask": self.s1_context().mask.copy()}

    def step(self, action: HierarchicalAction):
        self._require_runtime()
        if self._current_request is None or self._request_context is None:
            return self._terminal_observation(), 0.0, True, False, {"elapsed_minutes": 0.0, "events": []}
        if not isinstance(action, HierarchicalAction):
            invalid = HierarchicalAction(-1, -1, None)
            raise InvalidHierarchicalActionError(
                self._current_request,
                invalid,
                f"received action={action!r}; expected HierarchicalAction",
            )
        request = self._current_request
        plan = self._materialize_action(action)
        start_time = float(self.simulator.clock)
        start_metrics = self.simulator.interval_metrics_snapshot()
        self._events = []
        try:
            self._accept_plan(request, plan)
        except ProjectionUnavailableError as exc:
            raise InvalidHierarchicalActionError(request, action, str(exc)) from exc
        self._request_index += 1
        if self._request_index < len(self._requests):
            self._select_next_request()
            self._advance_to(float(self._current_request.decision_time))
            terminated = False
            observation = self._build_observation(action)
        else:
            self._current_request = None
            self._request_context = None
            self._drain_pending()
            terminated = True
            observation = self._terminal_observation()
        delta = self.simulator.interval_metrics_delta(start_metrics)
        reward = interval_reward(
            delta,
            float(plan.detour_ratio),
            plan.s2 is not None,
            self.config.reward_weights,
            self.config.reward_scales,
        )
        if not np.isfinite(reward):
            raise NonFiniteRewardError(f"non-finite reward for vehicle={request.vehicle_id}, action={action}")
        info = self._build_info(action, plan, delta, float(self.simulator.clock) - start_time)
        return observation, float(reward), bool(terminated), False, info

    def s1_context(self):
        if self._request_context is None:
            raise RuntimeError("no active request context")
        return self.generator.build_s1_context(self._request_context)

    def s2_context(self, s1_index: int):
        if self._request_context is None:
            raise RuntimeError("no active request context")
        return self.generator.build_s2_context(self._request_context, int(s1_index))

    def lambda_context(self, s1_index: int, s2_index: int):
        if self._request_context is None:
            raise RuntimeError("no active request context")
        return self.generator.build_lambda_context(self._request_context, int(s1_index), int(s2_index))

    def _consume_episode(self) -> EpisodeData:
        if not self._used_initial_episode:
            self._used_initial_episode = True
            return self._initial_episode
        return self.episode_factory()

    def _select_next_request(self) -> None:
        if self._request_index >= len(self._requests):
            self._current_request = None
            self._request_context = None
            return
        self._current_request = self._requests[self._request_index]
        self._request_context = self.generator.build_request_context(self._current_request)

    def _materialize_action(self, action: HierarchicalAction) -> ChargingPlan:
        request = self._current_request
        context = self._request_context
        assert request is not None and context is not None
        raw_indices = (action.s1_index, action.s2_index)
        lambda_valid = action.lambda_index is None or (
            isinstance(action.lambda_index, Integral)
            and not isinstance(action.lambda_index, bool)
        )
        if (
            any(
                not isinstance(value, Integral) or isinstance(value, bool)
                for value in raw_indices
            )
            or not lambda_valid
        ):
            raise InvalidHierarchicalActionError(
                request,
                HierarchicalAction(-1, -1, None),
                f"action indices must be non-boolean integers; received {action!r}",
            )
        try:
            s1_index, s2_index = int(action.s1_index), int(action.s2_index)
            s1 = self.generator.build_s1_context(context)
            if not 0 <= s1_index < 72 or not s1.mask[s1_index]:
                raise ValueError("s1 is out of range or masked")
            s2 = self.generator.build_s2_context(context, s1_index)
            if not 0 <= s2_index <= 72 or not s2.mask[s2_index]:
                raise ValueError("s2 is out of range or masked")
            if s2_index == 72:
                if action.lambda_index is not None:
                    raise ValueError("single-stop action must omit lambda index")
            else:
                if action.lambda_index is None:
                    raise ValueError("split action requires lambda index")
                lambdas = self.generator.build_lambda_context(context, s1_index, s2_index)
                lambda_index = int(action.lambda_index)
                if lambdas is None or not 0 <= lambda_index < 15 or not lambdas.mask[lambda_index]:
                    raise ValueError("lambda is out of range or masked")
            return self.generator.materialize_plan(context, action)
        except InvalidHierarchicalActionError:
            raise
        except (IndexError, TypeError, ValueError) as exc:
            raise InvalidHierarchicalActionError(request, action, str(exc)) from exc

    def _accept_plan(self, request: VehicleRequest, plan: ChargingPlan) -> None:
        first_leg = plan.route.legs[0]
        first_arrival = float(request.decision_time) + self._path_time(first_leg, float(request.decision_time))
        first_soc = float(request.vehicle_spec.initial_soc) - self._path_energy(first_leg, float(request.decision_time), request) / float(request.vehicle_spec.battery_capacity)
        first_kwh = max(0.0, (float(plan.lambda1) - first_soc) * float(request.vehicle_spec.battery_capacity))
        first_charge_request = ChargingSocRequest(
            vehicle_id=int(request.vehicle_id),
            station_id=int(plan.s1),
            arrival_time=float(first_arrival),
            vehicle_spec=request.vehicle_spec,
            arrival_soc=float(first_soc),
            target_soc=float(plan.lambda1),
        )
        second_arrival = None
        second_kwh = 0.0
        if plan.s2 is not None:
            provisional_departure = self.simulator.estimate_completion_time(
                first_charge_request
            )
            second_leg = plan.route.legs[1]
            second_arrival = provisional_departure + self._path_time(second_leg, provisional_departure)
            second_soc = float(plan.lambda1) - self._path_energy(second_leg, provisional_departure, request) / float(request.vehicle_spec.battery_capacity)
            second_kwh = max(0.0, (float(request.target_soc) - second_soc) * float(request.vehicle_spec.battery_capacity))
        self.incoming.add_plan(plan, first_arrival, first_kwh, second_arrival, second_kwh)
        self._pending[int(request.vehicle_id)] = _PendingPlan(request, plan)
        self._schedule_arrival(request, int(plan.s1), first_arrival, first_soc, float(plan.lambda1), 1)

    def _schedule_arrival(self, request, station_id, arrival_time, arrival_soc, target_soc, leg_index) -> None:
        self.simulator.schedule_soc_arrival(
            ChargingSocRequest(
                vehicle_id=int(request.vehicle_id), station_id=int(station_id),
                arrival_time=float(arrival_time), vehicle_spec=request.vehicle_spec,
                arrival_soc=float(arrival_soc), target_soc=float(target_soc),
            )
        )
        self._scheduled_leg_by_vehicle[int(request.vehicle_id)] = int(leg_index)

    def _advance_to(self, target_time: float) -> None:
        target = float(target_time)
        while True:
            clock = float(self.simulator.clock)
            due = self.simulator.next_scheduled_soc_arrival_time()
            if due is not None and float(due) <= clock + _EPS:
                for charge_request in self.simulator.pop_due_scheduled_soc_arrivals(clock):
                    vehicle_id = int(charge_request.vehicle_id)
                    leg_index = self._scheduled_leg_by_vehicle.pop(vehicle_id)
                    self.incoming.mark_arrived(vehicle_id, leg_index)
                    self._events.append({"type": "due_arrival", "time": clock, "vehicle_id": vehicle_id, "leg_index": leg_index})
                    self.simulator.enqueue_soc_arrival(charge_request)
                continue
            if clock >= target - _EPS:
                return
            next_time = target
            if due is not None:
                next_time = min(next_time, float(due))
            next_time = self.simulator.next_internal_event_time(next_time)
            self._events.append({"type": "advance", "from": clock, "time": next_time})
            for assignment in self.simulator.advance_to(next_time):
                self._handle_completion(assignment)

    def _handle_completion(self, assignment: ChargingAssignment) -> None:
        vehicle_id = int(assignment.vehicle_id)
        pending = self._pending.get(vehicle_id)
        if pending is None:
            return
        leg_index = int(pending.current_leg)
        self._events.append({"type": "completion", "time": float(assignment.end_time), "vehicle_id": vehicle_id, "leg_index": leg_index})
        if leg_index == 1 and pending.plan.s2 is not None:
            pending.current_leg = 2
            second_leg = pending.plan.route.legs[1]
            departure = float(assignment.end_time)
            arrival = departure + self._path_time(second_leg, departure)
            arrival_soc = float(pending.plan.lambda1) - self._path_energy(second_leg, departure, pending.request) / float(pending.request.vehicle_spec.battery_capacity)
            second_kwh = max(0.0, (float(pending.request.target_soc) - arrival_soc) * float(pending.request.vehicle_spec.battery_capacity))
            self.incoming.update_second_leg(vehicle_id, arrival, second_kwh)
            self._events.append({"type": "incoming_update", "time": departure, "vehicle_id": vehicle_id, "leg_index": 2})
            self._schedule_arrival(pending.request, int(pending.plan.s2), arrival, arrival_soc, float(pending.request.target_soc), 2)
        else:
            self._pending.pop(vehicle_id, None)

    def _drain_pending(self) -> None:
        while self._pending:
            clock = float(self.simulator.clock)
            before_pending = tuple(sorted(self._pending))
            before_scheduled = len(self.simulator.scheduled_soc_arrivals())
            due = self.simulator.next_scheduled_soc_arrival_time()
            limit = math.inf if due is None else float(due)
            internal = self.simulator.next_internal_event_time(limit)
            candidates = [
                value
                for value in (due, internal)
                if value is not None and math.isfinite(float(value))
            ]
            if not candidates:
                raise RuntimeError(
                    "terminal drain made no progress for pending vehicles "
                    f"{list(before_pending)}: no future arrival or charging event"
                )
            target = max(clock, min(float(value) for value in candidates))
            self._advance_to(target)
            after_scheduled = len(self.simulator.scheduled_soc_arrivals())
            if (
                float(self.simulator.clock) <= clock + _EPS
                and tuple(sorted(self._pending)) == before_pending
                and after_scheduled == before_scheduled
            ):
                raise RuntimeError(
                    "terminal drain made no progress for pending vehicles "
                    f"{list(before_pending)} at time={clock}"
                )

    def _path_time(self, leg, departure_time: float) -> float:
        candidate = self.episode.network if self.episode is not None else None
        function = getattr(candidate, "path_time", None) or getattr(self.oracle, "path_time", None)
        if not callable(function):
            raise ValueError("exact route path_time is unavailable")
        value = float(function(int(leg.source), int(leg.target), float(departure_time), route_nodes=tuple(leg.node_ids)))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("exact route path_time must be finite and nonnegative")
        return value

    def _path_energy(self, leg, departure_time: float, request: VehicleRequest) -> float:
        candidate = self.episode.network if self.episode is not None else None
        function = getattr(candidate, "path_energy", None)
        if callable(function):
            value = float(function(int(leg.source), int(leg.target), float(departure_time), request.vehicle_spec, route_nodes=tuple(leg.node_ids)))
        else:
            value = float(leg.distance_m) * float(request.vehicle_spec.rho_kwh_per_km) / 1000.0
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("exact route path_energy must be finite and nonnegative")
        return value

    def _build_observation(self, action=None) -> StructuredObservation:
        request, context = self._current_request, self._request_context
        assert request is not None and context is not None
        state = self.simulator.get_state(query_time=float(self.simulator.clock))
        state["network"] = self.episode.network
        observation = build_structured_observation(
            request=request, now=float(self.simulator.clock), simulator_state=state,
            station_specs=tuple(sorted(self.episode.station_specs, key=lambda item: item.station_id)),
            oracle=self.oracle, feasibility_context=context,
            incoming_summary=self.incoming.summarize(float(self.simulator.clock), self.config.incoming_windows_minutes),
            scales=self.config.observation_scales,
        )
        if not np.isfinite(observation.request).all() or not np.isfinite(observation.stations).all():
            raise NonFiniteObservationError(f"non-finite observation for vehicle={request.vehicle_id}, action={action}")
        return observation

    @staticmethod
    def _terminal_observation() -> StructuredObservation:
        return StructuredObservation(0, 0, np.zeros(16, dtype=np.float32), np.zeros((72, 33), dtype=np.float32))

    def _build_info(self, action, plan, delta: IntervalMetrics, elapsed_minutes: float) -> dict:
        scales = self.config.reward_scales
        raw = {
            "wait_vehicle_minutes": float(delta.wait_vehicle_minutes),
            "grid_energy_kwh": float(delta.grid_energy_kwh),
            "renewable_curtailed_kwh": float(delta.renewable_curtailed_kwh),
            "detour_ratio": float(plan.detour_ratio),
            "additional_stop": int(plan.s2 is not None),
        }
        normalized = {
            "wait_vehicle_minutes": raw["wait_vehicle_minutes"] / float(scales.wait_time_minutes),
            "grid_energy_kwh": raw["grid_energy_kwh"] / float(scales.grid_energy_kwh),
            "renewable_curtailed_kwh": raw["renewable_curtailed_kwh"] / float(scales.renewable_curtailment_kwh),
            "detour_ratio": raw["detour_ratio"] / float(scales.detour_ratio),
            "additional_stop": raw["additional_stop"] / float(scales.additional_stop),
        }
        info = {
            "action": action, "plan": plan, "raw_reward_components": raw,
            "normalized_reward_components": normalized, "elapsed_minutes": float(elapsed_minutes),
            "events": list(self._events),
        }
        if self._current_request is not None:
            info["s1_mask"] = self.s1_context().mask.copy()
        return info

    def _require_runtime(self) -> None:
        if self.episode is None or self.simulator is None:
            raise RuntimeError("reset() must be called before step()")


def _episode_checkpoint_state(episode: EpisodeData | None) -> dict[str, Any] | None:
    if episode is None:
        return None
    return {
        "station_specs": copy.deepcopy(episode.station_specs),
        "vehicle_requests": copy.deepcopy(episode.vehicle_requests),
        "timestep_minutes": float(episode.timestep_minutes),
        "initial_state": copy.deepcopy(episode.initial_state),
        "has_network": episode.network is not None,
    }


def _episode_from_checkpoint(
    state: Any, static_network: Any | None
) -> EpisodeData | None:
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise TypeError("episode checkpoint state must be a mapping")
    expected_keys = {
        "station_specs",
        "vehicle_requests",
        "timestep_minutes",
        "initial_state",
        "has_network",
    }
    if set(state) != expected_keys:
        raise ValueError("episode checkpoint state has an invalid schema")
    has_network = bool(state["has_network"])
    if has_network and static_network is None:
        raise ValueError("episode checkpoint requires the configured static network")
    return EpisodeData(
        station_specs=tuple(copy.deepcopy(state["station_specs"])),
        vehicle_requests=tuple(copy.deepcopy(state["vehicle_requests"])),
        network=static_network if has_network else None,
        timestep_minutes=float(state["timestep_minutes"]),
        initial_state=copy.deepcopy(state["initial_state"]),
    )
