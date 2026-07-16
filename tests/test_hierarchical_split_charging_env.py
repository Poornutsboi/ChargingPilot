from __future__ import annotations

import copy
import unittest

import numpy as np

from chargingpilot.environment.hierarchical_split_charging_env import (
    HierarchicalSplitChargingRequestEnv,
)
from chargingpilot.environment.models import EpisodeData, VehicleRequest
from chargingpilot.environment.structured_observation import StructuredObservation
from chargingpilot.routing import FeasiblePlanGenerator, HierarchicalAction, InvalidHierarchicalActionError
from chargingpilot.routing.models import (
    ChargingPlan,
    LambdaDecisionContext,
    RequestFeasibilityContext,
    RouteLeg,
    RouteResult,
    S1DecisionContext,
    S2DecisionContext,
    ServiceBaseline,
)
from chargingpilot.simulator.models import ChargingSocRequest, StationSpec, VehicleSpec
from chargingpilot.simulator.simulator import SimulatorCore
from chargingpilot.simulator.station import StationRuntime


def _vehicle() -> VehicleSpec:
    return VehicleSpec(
        battery_capacity=60.0,
        initial_soc=0.4,
        soc_min=0.0,
        p_max_kw=60.0,
        p_min_kw=10.0,
        rho_kwh_per_km=0.18,
        origin=0,
        destination=2,
        departure_time=0.0,
        path_nodes=(0, 1, 2),
        path_edges=("0->1", "1->2"),
    )


def _charge_request(vehicle_id: int, target_soc: float) -> ChargingSocRequest:
    return ChargingSocRequest(
        vehicle_id=vehicle_id,
        station_id=1,
        arrival_time=0.0,
        vehicle_spec=_vehicle(),
        arrival_soc=0.4,
        target_soc=target_soc,
    )


STATION_IDS = tuple(range(100, 172))
ORIGIN = 1
DESTINATION = 999


class _Network:
    def __init__(self) -> None:
        self.path_time_calls: list[tuple[int, int, float, tuple[int, ...]]] = []
        self.path_energy_calls: list[tuple[int, int, float, tuple[int, ...]]] = []

    def path_time(self, source, target, departure_time, route_nodes=None) -> float:
        nodes = tuple(route_nodes or (source, target))
        self.path_time_calls.append((source, target, departure_time, nodes))
        return 1.0

    def path_energy(self, source, target, departure_time, vehicle, route_nodes=None) -> float:
        nodes = tuple(route_nodes or (source, target))
        self.path_energy_calls.append((source, target, departure_time, nodes))
        return 5.0


def _single_route(station_id: int) -> RouteResult:
    legs = (
        RouteLeg(ORIGIN, station_id, (ORIGIN, 50, station_id), 1_000.0),
        RouteLeg(station_id, DESTINATION, (station_id, 60, DESTINATION), 1_000.0),
    )
    return RouteResult((station_id,), legs, (ORIGIN, 50, station_id, 60, DESTINATION), 2_000.0)


def _split_route() -> RouteResult:
    legs = (
        RouteLeg(ORIGIN, 100, (ORIGIN, 50, 100), 1_000.0),
        RouteLeg(100, 101, (100, 55, 101), 1_000.0),
        RouteLeg(101, DESTINATION, (101, 60, DESTINATION), 1_000.0),
    )
    return RouteResult((100, 101), legs, (ORIGIN, 50, 100, 55, 101, 60, DESTINATION), 3_000.0)


class _Oracle:
    def __init__(self, network: _Network) -> None:
        self.network = network
        self.station_ids = STATION_IDS
        self.node_ids = tuple(sorted((ORIGIN, 50, 55, 60, *STATION_IDS, DESTINATION)))
        self.node_to_index = {node: index for index, node in enumerate(self.node_ids)}
        self.station_to_index = {station: index for index, station in enumerate(STATION_IDS)}
        self.max_finite_distance_m = 10_000.0

    def service_baseline(self, origin: int, destination: int) -> ServiceBaseline:
        return ServiceBaseline(100, 2_000.0, (origin, 100, destination))

    def direct_route(self, origin: int, destination: int) -> RouteResult:
        leg = RouteLeg(origin, destination, (origin, destination), 1_500.0)
        return RouteResult((), (leg,), leg.node_ids, leg.distance_m)

    def route_via(self, origin: int, destination: int, required) -> RouteResult:
        required = tuple(required)
        if required == (100, 101):
            return _split_route()
        if len(required) == 1:
            return _single_route(required[0])
        raise ValueError("route unavailable")

    def path_time(self, *args, **kwargs) -> float:
        return self.network.path_time(*args, **kwargs)


class _ScriptedGenerator:
    def __init__(self, oracle: _Oracle) -> None:
        self.oracle = oracle
        self.station_ids = STATION_IDS
        self.none_index = len(STATION_IDS)

    def build_request_context(self, request: VehicleRequest) -> RequestFeasibilityContext:
        return RequestFeasibilityContext(
            request=request,
            baseline=self.oracle.service_baseline(ORIGIN, DESTINATION),
            station_ids=STATION_IDS,
            single_routes={station: _single_route(station) for station in STATION_IDS},
            split_routes={(0, 1): _split_route()},
            arrival_soc_s1={0: 0.75},
        )

    def build_s1_context(self, context) -> S1DecisionContext:
        mask = np.zeros(72, dtype=np.bool_)
        mask[0] = True
        return S1DecisionContext(context, mask)

    def build_s2_context(self, context, s1_index: int) -> S2DecisionContext:
        mask = np.zeros(73, dtype=np.bool_)
        routes: list[RouteResult | None] = [None] * 73
        if int(s1_index) == 0:
            mask[1] = True
            mask[72] = True
            routes[1] = _split_route()
            routes[72] = _single_route(100)
        return S2DecisionContext(
            context,
            int(s1_index),
            mask,
            np.zeros((73, 6), dtype=np.float32),
            tuple(routes),
        )

    def build_lambda_context(self, context, s1_index: int, s2_index: int):
        if (int(s1_index), int(s2_index)) != (0, 1):
            return None
        mask = np.zeros(15, dtype=np.bool_)
        mask[12] = True
        return LambdaDecisionContext(
            context,
            0,
            1,
            mask,
            np.zeros(5, dtype=np.float32),
            np.round(np.arange(0.30, 1.0001, 0.05), 2).astype(np.float32),
        )

    def materialize_plan(self, context, action: HierarchicalAction) -> ChargingPlan:
        if action == HierarchicalAction(0, 72, None):
            return ChargingPlan(
                context.request.vehicle_id,
                100,
                None,
                context.request.target_soc,
                context.baseline,
                _single_route(100),
                0.10,
            )
        if action == HierarchicalAction(0, 1, 12):
            return ChargingPlan(
                context.request.vehicle_id,
                100,
                101,
                0.90,
                context.baseline,
                _split_route(),
                0.20,
            )
        raise InvalidHierarchicalActionError(context.request, action, "scripted action is masked")


class _ChronologyNetwork(_Network):
    def path_time(self, source, target, departure_time, route_nodes=None) -> float:
        nodes = tuple(route_nodes or (source, target))
        self.path_time_calls.append((source, target, departure_time, nodes))
        return 2.0 if (int(source), int(target)) == (100, 101) else 0.0

    def path_energy(self, source, target, departure_time, vehicle, route_nodes=None) -> float:
        nodes = tuple(route_nodes or (source, target))
        self.path_energy_calls.append((source, target, departure_time, nodes))
        return 0.0


class _ChronologyGenerator(_ScriptedGenerator):
    def build_lambda_context(self, context, s1_index: int, s2_index: int):
        result = super().build_lambda_context(context, s1_index, s2_index)
        if result is None:
            return None
        mask = np.zeros(15, dtype=np.bool_)
        mask[3] = True
        return LambdaDecisionContext(
            context,
            0,
            1,
            mask,
            result.features.copy(),
            result.bins.copy(),
        )

    def materialize_plan(self, context, action: HierarchicalAction) -> ChargingPlan:
        if action == HierarchicalAction(0, 1, 3):
            return ChargingPlan(
                context.request.vehicle_id,
                100,
                101,
                0.45,
                context.baseline,
                _split_route(),
                0.20,
            )
        return super().materialize_plan(context, action)


def _policy_request(vehicle_id: int, decision_time: float = 0.0) -> VehicleRequest:
    spec = VehicleSpec(
        battery_capacity=100.0,
        initial_soc=0.80,
        soc_min=0.10,
        p_max_kw=60.0,
        p_min_kw=10.0,
        rho_kwh_per_km=0.20,
        origin=ORIGIN,
        destination=DESTINATION,
        departure_time=decision_time,
        path_nodes=(ORIGIN, DESTINATION),
        path_edges=(),
    )
    return VehicleRequest(vehicle_id, decision_time, spec, 0.90)


def _make_environment(requests: tuple[VehicleRequest, ...]):
    network = _Network()
    oracle = _Oracle(network)
    specs = tuple(
        StationSpec(
            station_id=station_id,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            p_grid_max_kw=60.0,
            eta=1.0,
        )
        for station_id in STATION_IDS
    )
    episode = EpisodeData(specs, requests, network=network, timestep_minutes=1.0)
    environment = HierarchicalSplitChargingRequestEnv(
        episode_factory=lambda: episode,
        oracle=oracle,
        plan_generator=_ScriptedGenerator(oracle),
    )
    return environment, network


def _make_real_generator_environment(requests: tuple[VehicleRequest, ...]):
    network = _Network()
    oracle = _Oracle(network)
    specs = tuple(
        StationSpec(
            station_id=station_id,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            p_grid_max_kw=60.0,
            eta=1.0,
        )
        for station_id in STATION_IDS
    )
    episode = EpisodeData(specs, requests, network=network, timestep_minutes=1.0)
    environment = HierarchicalSplitChargingRequestEnv(
        episode_factory=lambda: episode,
        oracle=oracle,
        plan_generator=FeasiblePlanGenerator(oracle),
    )
    return environment


def _make_small_timestep_drain_environment(grid_power_kw: float):
    network = _ChronologyNetwork()
    oracle = _Oracle(network)
    specs = tuple(
        StationSpec(
            station_id=station_id,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            p_grid_max_kw=grid_power_kw,
            eta=1.0,
        )
        for station_id in STATION_IDS
    )
    vehicle = VehicleSpec(
        battery_capacity=60.0,
        initial_soc=0.40,
        soc_min=0.10,
        p_max_kw=60.0,
        p_min_kw=10.0,
        rho_kwh_per_km=0.20,
        origin=ORIGIN,
        destination=DESTINATION,
        departure_time=0.0,
        path_nodes=(ORIGIN, DESTINATION),
        path_edges=(),
    )
    episode = EpisodeData(
        specs,
        (VehicleRequest(1, 0.0, vehicle, 0.45),),
        network=network,
        timestep_minutes=0.0001,
    )
    environment = HierarchicalSplitChargingRequestEnv(
        episode_factory=lambda: episode,
        oracle=oracle,
        plan_generator=_ScriptedGenerator(oracle),
    )
    environment.reset()
    return environment


class HierarchicalSplitChargingEnvironmentTests(unittest.TestCase):
    def test_real_generator_checkpoint_restore_preserves_next_transition(self) -> None:
        requests = (_policy_request(1, 0.0), _policy_request(2, 10.0))
        environment = _make_real_generator_environment(requests)
        environment.reset()
        action = HierarchicalAction(0, 72, None)
        environment.step(action)
        checkpoint_state = copy.deepcopy(environment.state_dict())

        restored = _make_real_generator_environment(requests)
        restored.load_state_dict(checkpoint_state)

        expected = environment.step(action)
        actual = restored.step(action)

        np.testing.assert_array_equal(actual[0].request, expected[0].request)
        np.testing.assert_array_equal(actual[0].stations, expected[0].stations)
        self.assertEqual(actual[1:4], expected[1:4])
        self.assertEqual(actual[4]["plan"], expected[4]["plan"])
        self.assertEqual(
            actual[4]["raw_reward_components"],
            expected[4]["raw_reward_components"],
        )

    def test_checkpoint_state_rebuilds_module_network_and_preserves_next_transition(
        self,
    ) -> None:
        requests = (_policy_request(1, 0.0), _policy_request(2, 10.0))
        environment, network = _make_environment(requests)
        network.nonserializable_module = np
        environment.reset()
        environment.step(HierarchicalAction(0, 72, None))

        checkpoint_state = copy.deepcopy(environment.state_dict())

        restored, restored_network = _make_environment(requests)
        restored_network.nonserializable_module = np
        restored.load_state_dict(checkpoint_state)
        self.assertIs(restored.episode.network, restored_network)
        self.assertIs(restored._initial_episode.network, restored_network)
        self.assertEqual(restored.current_vehicle_id, environment.current_vehicle_id)
        self.assertEqual(restored.current_time, environment.current_time)
        self.assertEqual(restored.incoming.snapshot(), environment.incoming.snapshot())

        expected = environment.step(HierarchicalAction(0, 72, None))
        actual = restored.step(HierarchicalAction(0, 72, None))

        np.testing.assert_array_equal(actual[0].request, expected[0].request)
        np.testing.assert_array_equal(actual[0].stations, expected[0].stations)
        self.assertEqual(actual[1:4], expected[1:4])
        self.assertEqual(actual[4]["plan"], expected[4]["plan"])
        self.assertEqual(
            actual[4]["raw_reward_components"],
            expected[4]["raw_reward_components"],
        )

    def test_completion_projection_rejects_permanently_zero_power(self) -> None:
        spec = StationSpec(
            station_id=1,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            p_grid_max_kw=0.0,
            eta=1.0,
        )
        simulator = SimulatorCore(
            [spec], timestep_minutes=1.0, exact_internal_events=True
        )
        candidate = _charge_request(2, 0.50)

        with self.assertRaisesRegex(
            RuntimeError, r"projection unavailable.*station=1.*vehicle=2"
        ):
            simulator.estimate_completion_time(candidate)

    def test_projection_failure_is_contextual_invalid_action(self) -> None:
        environment = _make_small_timestep_drain_environment(0.0)
        action = HierarchicalAction(0, 1, 12)

        with self.assertRaises(InvalidHierarchicalActionError) as raised:
            environment.step(action)

        message = str(raised.exception)
        self.assertIn("vehicle=1", message)
        self.assertIn("action=(0,1,12)", message)
        self.assertIn("projection unavailable", message)

    def test_zero_energy_request_completes_immediately_at_arrival(self) -> None:
        spec = StationSpec(
            station_id=1,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            eta=1.0,
        )
        simulator = SimulatorCore(
            [spec], timestep_minutes=10.0, exact_internal_events=True
        )
        request = ChargingSocRequest(
            vehicle_id=1,
            station_id=1,
            arrival_time=0.0,
            vehicle_spec=_vehicle(),
            arrival_soc=0.50,
            target_soc=0.50,
        )

        self.assertEqual(simulator.estimate_completion_time(request), 0.0)
        simulator.enqueue_soc_arrival(request)
        assignments = simulator.advance_to(0.0)

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].start_time, 0.0)
        self.assertEqual(assignments[0].end_time, 0.0)
        self.assertEqual(assignments[0].energy_delivered_kwh, 0.0)
        with self.assertRaisesRegex(ValueError, "target_soc must be >= arrival_soc"):
            ChargingSocRequest(
                vehicle_id=2,
                station_id=1,
                arrival_time=0.0,
                vehicle_spec=_vehicle(),
                arrival_soc=0.50,
                target_soc=0.49,
            )

    def test_terminal_drain_is_not_limited_by_timestep_iteration_count(self) -> None:
        environment = _make_small_timestep_drain_environment(60.0)

        _observation, _reward, terminated, truncated, _info = environment.step(
            HierarchicalAction(0, 72, None)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(environment.current_time, 3.0)

    def test_terminal_drain_detects_pending_state_with_no_future_progress(self) -> None:
        environment = _make_small_timestep_drain_environment(0.0)

        with self.assertRaisesRegex(RuntimeError, r"no progress.*vehicle.*1"):
            environment.step(HierarchicalAction(0, 72, None))

    def test_completion_projection_uses_queue_energy_and_efficiency_without_mutation(self) -> None:
        spec = StationSpec(
            station_id=1,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            p_grid_max_kw=60.0,
            eta=0.5,
        )
        simulator = SimulatorCore(
            [spec], timestep_minutes=10.0, exact_internal_events=True
        )
        blocker = _charge_request(1, 0.45)
        candidate = _charge_request(2, 0.50)
        simulator.enqueue_soc_arrival(blocker)
        before = simulator.get_state(vehicle_info=True)

        completion_time = simulator.estimate_completion_time(candidate)

        self.assertAlmostEqual(completion_time, 18.0)
        self.assertEqual(simulator.get_state(vehicle_info=True), before)

    def test_second_leg_is_processed_chronologically_inside_long_interval(self) -> None:
        network = _ChronologyNetwork()
        oracle = _Oracle(network)
        specs = tuple(
            StationSpec(
                station_id=station_id,
                charge_capacity=1,
                p_plug_kw=60.0,
                p_max_kw=60.0,
                p_grid_max_kw=60.0,
                eta=1.0,
            )
            for station_id in STATION_IDS
        )
        vehicle = VehicleSpec(
            battery_capacity=60.0,
            initial_soc=0.40,
            soc_min=0.10,
            p_max_kw=60.0,
            p_min_kw=10.0,
            rho_kwh_per_km=0.20,
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=0.0,
            path_nodes=(ORIGIN, DESTINATION),
            path_edges=(),
        )
        requests = (
            VehicleRequest(1, 0.0, vehicle, 0.50),
            VehicleRequest(2, 10.0, vehicle, 0.50),
        )
        episode = EpisodeData(specs, requests, network=network, timestep_minutes=10.0)
        environment = HierarchicalSplitChargingRequestEnv(
            episode_factory=lambda: episode,
            oracle=oracle,
            plan_generator=_ChronologyGenerator(oracle),
        )
        environment.reset()
        environment.simulator.enqueue_soc_arrival(
            ChargingSocRequest(
                vehicle_id=999,
                station_id=101,
                arrival_time=0.0,
                vehicle_spec=vehicle,
                arrival_soc=0.40,
                target_soc=0.50,
            )
        )

        _observation, _reward, terminated, _truncated, info = environment.step(
            HierarchicalAction(0, 1, 3)
        )

        self.assertFalse(terminated)
        second_arrivals = [
            event
            for event in info["events"]
            if event["type"] == "due_arrival" and event["leg_index"] == 2
        ]
        self.assertEqual(len(second_arrivals), 1)
        self.assertAlmostEqual(second_arrivals[0]["time"], 5.0)
        self.assertAlmostEqual(
            info["raw_reward_components"]["wait_vehicle_minutes"], 1.0
        )
        self.assertAlmostEqual(info["raw_reward_components"]["grid_energy_kwh"], 12.0)
        station_two = environment.simulator.get_state()["stations"][101]
        self.assertEqual(station_two["active_vehicle_ids"], [])

    def test_mid_timestep_completion_releases_queue_at_exact_event_time(self) -> None:
        spec = StationSpec(
            station_id=1,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            p_grid_max_kw=60.0,
            eta=1.0,
        )
        station = StationRuntime(
            spec, timestep_minutes=10.0, exact_internal_events=True
        )
        station.enqueue_soc_request(_charge_request(1, 0.45))
        station.enqueue_soc_request(_charge_request(2, 0.45))

        station.advance_to(10.0)
        assignments = station.drain_completed_assignments()

        self.assertEqual([item.vehicle_id for item in assignments], [1, 2])
        self.assertAlmostEqual(assignments[0].end_time, 3.0)
        self.assertAlmostEqual(assignments[1].start_time, 3.0)
        self.assertAlmostEqual(assignments[1].end_time, 6.0)
        self.assertAlmostEqual(assignments[1].wait_time, 3.0)
        self.assertAlmostEqual(station.interval_metrics().wait_vehicle_minutes, 3.0)
        self.assertAlmostEqual(station.interval_metrics().grid_energy_kwh, 6.0)

    def test_wait_vehicle_minutes_are_exact_cumulative_and_snapshot_preserved(self) -> None:
        spec = StationSpec(
            station_id=1,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_max_kw=60.0,
            eta=1.0,
        )
        simulator = SimulatorCore([spec], timestep_minutes=1.0)
        simulator.enqueue_soc_arrival(_charge_request(1, 0.505))
        simulator.enqueue_soc_arrival(_charge_request(2, 0.45))
        start = simulator.interval_metrics_snapshot()

        simulator.advance_to(4.0)
        middle = simulator.interval_metrics_snapshot()
        self.assertAlmostEqual(
            simulator.interval_metrics_delta(start).wait_vehicle_minutes,
            4.0,
        )

        simulator.advance_to(7.0)
        self.assertAlmostEqual(
            simulator.interval_metrics_delta(middle).wait_vehicle_minutes,
            3.0,
        )
        self.assertAlmostEqual(
            simulator.interval_metrics_snapshot().wait_vehicle_minutes,
            7.0,
        )

        station = StationRuntime(spec, timestep_minutes=1.0)
        station.enqueue_soc_request(_charge_request(1, 0.505))
        station.enqueue_soc_request(_charge_request(2, 0.45))
        station.advance_to(4.0)
        snapshot = station.snapshot()
        restored = StationRuntime(spec, timestep_minutes=1.0)
        restored.restore(snapshot)
        restored.advance_to(7.0)
        self.assertAlmostEqual(restored.interval_metrics().wait_vehicle_minutes, 7.0)

    def test_reset_schema_is_finite_and_excludes_future_requests(self) -> None:
        one, _ = _make_environment((_policy_request(1),))
        many, _ = _make_environment((_policy_request(1), _policy_request(99, 500.0)))

        observation, info = one.reset(seed=7)
        future_observation, _ = many.reset(seed=7)

        self.assertIsInstance(observation, StructuredObservation)
        self.assertIsInstance(info, dict)
        np.testing.assert_array_equal(observation.request, future_observation.request)
        np.testing.assert_array_equal(observation.stations, future_observation.stations)
        self.assertTrue(np.isfinite(observation.request).all())
        self.assertTrue(np.isfinite(observation.stations).all())

        blocker = ChargingSocRequest(
            vehicle_id=500,
            station_id=100,
            arrival_time=0.0,
            vehicle_spec=_policy_request(500).vehicle_spec,
            arrival_soc=0.80,
            target_soc=0.86,
        )
        one.simulator.enqueue_soc_arrival(blocker)
        action = HierarchicalAction(0, 1, 12)
        plan = one._materialize_action(action)
        first_arrival = 1.0
        first_soc = 0.75
        first_charge = ChargingSocRequest(
            vehicle_id=1,
            station_id=100,
            arrival_time=first_arrival,
            vehicle_spec=one._current_request.vehicle_spec,
            arrival_soc=first_soc,
            target_soc=0.90,
        )
        expected_second_arrival = one.simulator.estimate_completion_time(first_charge) + 1.0

        one._accept_plan(one._current_request, plan)

        second_record = next(
            record
            for record in one.incoming.snapshot()["records"]
            if record["leg_index"] == 2
        )
        self.assertAlmostEqual(
            second_record["expected_arrival_time"], expected_second_arrival
        )

    def test_invalid_action_error_contains_vehicle_and_action_context(self) -> None:
        environment, _ = _make_environment((_policy_request(17),))
        environment.reset()

        with self.assertRaisesRegex(
            InvalidHierarchicalActionError,
            r"vehicle=17.*action=\(1,72,None\)",
        ):
            environment.step(HierarchicalAction(1, 72, None))

        with self.assertRaisesRegex(
            InvalidHierarchicalActionError,
            r"vehicle=17.*received action='not-an-action'",
        ):
            environment.step("not-an-action")

        invalid_indices = (
            HierarchicalAction(0.9, 72, None),
            HierarchicalAction(False, 72, None),
            HierarchicalAction("0", 72, None),
            HierarchicalAction(0, "72", None),
            HierarchicalAction(0, 1, 12.9),
        )
        for action in invalid_indices:
            with self.subTest(action=action):
                with self.assertRaises(InvalidHierarchicalActionError) as raised:
                    environment.step(action)
                self.assertIn(repr(action), str(raised.exception))

    def test_materialized_exact_route_nodes_are_used_for_travel(self) -> None:
        environment, network = _make_environment((_policy_request(1),))
        environment.reset()
        network.path_time_calls.clear()
        network.path_energy_calls.clear()

        environment.step(HierarchicalAction(0, 72, None))

        self.assertIn((ORIGIN, 100, 0.0, (ORIGIN, 50, 100)), network.path_time_calls)
        self.assertIn((ORIGIN, 100, 0.0, (ORIGIN, 50, 100)), network.path_energy_calls)

    def test_due_arrival_advance_completion_and_incoming_update_are_ordered(self) -> None:
        environment, _ = _make_environment((_policy_request(1),))
        environment.reset()

        _observation, _reward, terminated, truncated, info = environment.step(
            HierarchicalAction(0, 1, 12)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        event_names = [event["type"] for event in info["events"]]
        first_arrival = event_names.index("due_arrival")
        first_advance = event_names.index("advance", first_arrival + 1)
        first_completion = event_names.index("completion", first_advance + 1)
        incoming_update = event_names.index("incoming_update", first_completion + 1)
        self.assertLess(first_arrival, first_advance)
        self.assertLess(first_advance, first_completion)
        self.assertLess(first_completion, incoming_update)
        self.assertEqual(environment.incoming.summarize(environment.current_time).counts.sum(), 0.0)

    def test_next_request_transition_and_terminal_drain_include_tail_costs(self) -> None:
        environment, _ = _make_environment((_policy_request(1), _policy_request(2)))
        first_observation, _ = environment.reset()

        next_observation, first_reward, terminated, truncated, first_info = environment.step(
            HierarchicalAction(0, 72, None)
        )
        self.assertIsInstance(first_observation, StructuredObservation)
        self.assertIsInstance(next_observation, StructuredObservation)
        self.assertIsInstance(first_reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(environment.current_vehicle_id, 2)
        self.assertEqual(first_info["elapsed_minutes"], 0.0)

        terminal, reward, terminated, truncated, info = environment.step(
            HierarchicalAction(0, 72, None)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(info["raw_reward_components"]["wait_vehicle_minutes"], 0.0)
        self.assertGreater(info["raw_reward_components"]["grid_energy_kwh"], 0.0)
        self.assertLess(reward, 0.0)
        self.assertTrue(np.isfinite(terminal.request).all())
        self.assertTrue(np.isfinite(terminal.stations).all())
        np.testing.assert_array_equal(terminal.request, np.zeros(16, dtype=np.float32))
        np.testing.assert_array_equal(terminal.stations, np.zeros((72, 33), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
