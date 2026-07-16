from __future__ import annotations

import copy
from unittest import TestCase

import numpy as np

from chargingpilot.environment.models import VehicleRequest
from chargingpilot.routing import (
    FeasiblePlanGenerator,
    HierarchicalAction,
    InvalidHierarchicalActionError,
    NoFeasibleChargingPlanError,
    RouteLeg,
    RouteResult,
    ServiceBaseline,
)
from chargingpilot.simulator.models import VehicleSpec


ORIGIN = 1
DESTINATION = 9


def request(
    *,
    vehicle_id: int = 17,
    initial_soc: float = 0.50,
    target_soc: float = 0.80,
    soc_min: float = 0.10,
    battery_capacity: float = 100.0,
    rho_kwh_per_km: float = 0.10,
) -> VehicleRequest:
    return VehicleRequest(
        vehicle_id=vehicle_id,
        decision_time=0.0,
        vehicle_spec=VehicleSpec(
            battery_capacity=battery_capacity,
            initial_soc=initial_soc,
            soc_min=soc_min,
            p_max_kw=150.0,
            p_min_kw=0.0,
            rho_kwh_per_km=rho_kwh_per_km,
            origin=ORIGIN,
            destination=DESTINATION,
            departure_time=0.0,
            path_nodes=(ORIGIN, DESTINATION),
            path_edges=(),
        ),
        target_soc=target_soc,
    )


def route(required: tuple[int, ...], leg_distances_m: tuple[float, ...]) -> RouteResult:
    points = (ORIGIN, *required, DESTINATION)
    legs = tuple(
        RouteLeg(source, target, (source, target), float(distance))
        for source, target, distance in zip(points, points[1:], leg_distances_m)
    )
    return RouteResult(
        required_station_ids=required,
        legs=legs,
        node_ids=points,
        distance_m=float(sum(leg_distances_m)),
    )


class FakeOracle:
    def __init__(
        self,
        station_ids: tuple[int, ...],
        routes: dict[tuple[int, ...], RouteResult],
        *,
        baseline_distance_m: float = 100_000.0,
    ) -> None:
        self.station_ids = station_ids
        self.routes = routes
        self.baseline = ServiceBaseline(
            station_id=station_ids[0],
            distance_m=baseline_distance_m,
            node_ids=(ORIGIN, station_ids[0], DESTINATION),
        )
        self.baseline_calls = 0
        self.route_calls: list[tuple[int, ...]] = []

    def service_baseline(self, origin: int, destination: int) -> ServiceBaseline:
        if (origin, destination) != (ORIGIN, DESTINATION):
            raise AssertionError("unexpected OD")
        self.baseline_calls += 1
        return self.baseline

    def route_via(
        self,
        origin: int,
        destination: int,
        station_ids: tuple[int, ...],
    ) -> RouteResult:
        if (origin, destination) != (ORIGIN, DESTINATION):
            raise AssertionError("unexpected OD")
        required = tuple(station_ids)
        self.route_calls.append(required)
        try:
            return self.routes[required]
        except KeyError as exc:
            raise ValueError(f"no path via {required}") from exc


class FeasiblePlanGeneratorTests(TestCase):
    def test_find_first_feasible_action_prefers_baseline_single_and_matches_full_mask(
        self,
    ) -> None:
        station_ids = (2, 5)
        oracle = FakeOracle(
            station_ids,
            {
                (2,): route((2,), (20_000.0, 40_000.0)),
                (5,): route((5,), (20_000.0, 40_000.0)),
            },
        )
        oracle.baseline = ServiceBaseline(5, 100_000.0, (ORIGIN, 5, DESTINATION))
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        action = generator.find_first_feasible_action(context)
        plan = generator.materialize_plan(context, action)
        full_s1 = generator.build_s1_context(context)
        full_s2 = generator.build_s2_context(context, action.s1_index)

        self.assertEqual(action, HierarchicalAction(1, generator.none_index, None))
        self.assertEqual((plan.s1, plan.s2), (5, None))
        self.assertTrue(full_s1.mask[action.s1_index])
        self.assertTrue(full_s2.mask[action.s2_index])

    def test_find_first_feasible_action_returns_first_split_lambda_in_full_masks(
        self,
    ) -> None:
        oracle = FakeOracle(
            (2, 3),
            {(2, 3): route((2, 3), (20_000.0, 20_000.0, 20_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        action = generator.find_first_feasible_action(context)
        plan = generator.materialize_plan(context, action)
        full_s1 = generator.build_s1_context(context)
        full_s2 = generator.build_s2_context(context, action.s1_index)
        full_lambda = generator.build_lambda_context(
            context, action.s1_index, action.s2_index
        )

        self.assertEqual(action, HierarchicalAction(0, 1, 4))
        self.assertEqual((plan.s1, plan.s2, plan.lambda1), (2, 3, 0.5))
        self.assertTrue(full_s1.mask[action.s1_index])
        self.assertTrue(full_s2.mask[action.s2_index])
        self.assertTrue(full_lambda.mask[action.lambda_index])

    def test_find_first_feasible_action_matches_full_mask_no_plan(self) -> None:
        generator = FeasiblePlanGenerator(FakeOracle((2, 3), {}))
        context = generator.build_request_context(request())

        action = generator.find_first_feasible_action(context)

        self.assertIsNone(action)
        with self.assertRaises(NoFeasibleChargingPlanError):
            generator.build_s1_context(context)

    def test_register_restored_context_makes_it_canonical_and_reuses_routes(self) -> None:
        exact_route = route((2,), (20_000.0, 40_000.0))
        source = FeasiblePlanGenerator(FakeOracle((2,), {(2,): exact_route}))
        source_context = source.build_request_context(request())
        source.build_s2_context(source_context, 0)
        restored_context = copy.deepcopy(source_context)

        oracle = FakeOracle((2,), {(2,): exact_route})
        generator = FeasiblePlanGenerator(oracle)
        registered = generator.register_context(restored_context)

        self.assertIs(registered, restored_context)
        self.assertIs(generator.build_request_context(request()), restored_context)
        generator.build_s2_context(restored_context, 0)
        self.assertEqual(oracle.route_calls, [])
        with self.assertRaisesRegex(ValueError, "different canonical context"):
            generator.register_context(copy.deepcopy(restored_context))

    def test_rejects_unsorted_station_ids(self) -> None:
        oracle = FakeOracle((5, 2), {})

        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            FeasiblePlanGenerator(oracle)

    def test_sorted_station_ids_have_stable_indices_and_cached_routes(self) -> None:
        station_ids = (2, 5, 8)
        exact_routes = {
            (station_id,): route((station_id,), (20_000.0, 40_000.0))
            for station_id in station_ids
        }
        oracle = FakeOracle(station_ids, exact_routes)
        generator = FeasiblePlanGenerator(oracle)

        request_context = generator.build_request_context(request())
        first = generator.build_s1_context(request_context)
        calls_after_first_build = list(oracle.route_calls)
        second = generator.build_s1_context(request_context)

        self.assertEqual(request_context.station_ids, station_ids)
        np.testing.assert_array_equal(first.mask, np.ones(3, dtype=np.bool_))
        np.testing.assert_array_equal(second.mask, first.mask)
        self.assertEqual(generator.none_index, 3)
        self.assertEqual(oracle.baseline_calls, 1)
        self.assertEqual(oracle.route_calls, calls_after_first_build)
        self.assertEqual(
            [required for required in oracle.route_calls if len(required) == 1],
            [(2,), (5,), (8,)],
        )

    def _assert_single_detour(self, distance_m: float, expected: bool) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (20_000.0, distance_m - 20_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request(battery_capacity=1_000.0))

        try:
            actual = bool(generator.build_s1_context(context).mask[0])
        except NoFeasibleChargingPlanError as exc:
            actual = bool(exc.mask[0])

        self.assertEqual(actual, expected)

    def test_detour_just_below_boundary_is_valid(self) -> None:
        baseline = 100_000.0
        eps = max(1e-6, 1e-9 * baseline)
        self._assert_single_detour(1.6 * baseline - eps, True)

    def test_detour_one_epsilon_above_boundary_is_valid(self) -> None:
        baseline = 100_000.0
        eps = max(1e-6, 1e-9 * baseline)
        self._assert_single_detour(1.6 * baseline + eps, True)

    def test_detour_two_epsilons_above_boundary_is_invalid(self) -> None:
        baseline = 100_000.0
        eps = max(1e-6, 1e-9 * baseline)
        self._assert_single_detour(1.6 * baseline + 2.0 * eps, False)

    def test_unreachable_s1_is_masked_and_no_feasible_error_has_context(self) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (500_000.0, 100_000.0))},
            baseline_distance_m=1_000_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        with self.assertRaises(NoFeasibleChargingPlanError) as caught:
            generator.build_s1_context(context)

        error = caught.exception
        np.testing.assert_array_equal(error.mask, np.zeros(1, dtype=np.bool_))
        self.assertEqual(error.vehicle_id, 17)
        self.assertEqual((error.origin, error.destination), (ORIGIN, DESTINATION))
        self.assertEqual(error.initial_soc, 0.50)
        self.assertEqual(error.target_soc, 0.80)

    def test_unreachable_s2_is_masked(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {
                (2,): route((2,), (20_000.0, 800_000.0)),
                (2, 3): route((2, 3), (20_000.0, 800_000.0, 100_000.0)),
            },
            baseline_distance_m=1_000_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        s2_context = generator.build_s2_context(context, 0)

        self.assertFalse(s2_context.mask[1])
        self.assertFalse(s2_context.mask[generator.none_index])

    def test_target_soc_that_cannot_reach_destination_is_masked(self) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (20_000.0, 800_000.0))},
            baseline_distance_m=1_000_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        with self.assertRaises(NoFeasibleChargingPlanError) as caught:
            generator.build_s1_context(context)

        self.assertFalse(caught.exception.mask[0])

    def test_single_stop_requires_positive_s1_charge(self) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (0.0, 20_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(
            request(initial_soc=0.80, target_soc=0.80)
        )

        with self.assertRaises(NoFeasibleChargingPlanError) as caught:
            generator.build_s1_context(context)

        self.assertFalse(caught.exception.mask[0])

    def test_split_stop_requires_positive_s2_charge(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {(2, 3): route((2, 3), (20_000.0, 0.0, 20_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        lambda_context = generator.build_lambda_context(context, 0, 1)

        self.assertIsNotNone(lambda_context)
        self.assertFalse(lambda_context.mask[-5:].any())
        self.assertFalse(lambda_context.mask[-1])

    def test_split_uses_its_exact_first_leg_arrival_soc(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {
                (2,): route((2,), (20_000.0, 40_000.0)),
                (2, 3): route((2, 3), (450_000.0, 100_000.0, 20_000.0)),
            },
            baseline_distance_m=1_000_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        s2_context = generator.build_s2_context(context, 0)
        lambda_context = generator.build_lambda_context(context, 0, 1)

        self.assertAlmostEqual(context.arrival_soc_s1[0], 0.48)
        self.assertAlmostEqual(float(lambda_context.features[0]), 0.05)
        self.assertFalse(lambda_context.mask.any())
        self.assertFalse(s2_context.mask[1])

    def test_true_parent_bits_always_have_true_child_bits(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {
                (2,): route((2,), (20_000.0, 40_000.0)),
                (3,): route((3,), (300_000.0, 40_000.0)),
                (2, 3): route((2, 3), (20_000.0, 100_000.0, 20_000.0)),
            },
            baseline_distance_m=200_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        s1_context = generator.build_s1_context(context)

        for s1_index in np.flatnonzero(s1_context.mask):
            s2_context = generator.build_s2_context(context, int(s1_index))
            self.assertTrue(s2_context.mask.any())
            for s2_index in np.flatnonzero(s2_context.mask[:-1]):
                lambda_context = generator.build_lambda_context(
                    context, int(s1_index), int(s2_index)
                )
                self.assertIsNotNone(lambda_context)
                self.assertTrue(lambda_context.mask.any())

        self.assertTrue(generator.build_s2_context(context, 0).mask[1])

    def test_materialize_and_validate_reuse_the_stored_exact_route(self) -> None:
        exact = route((2, 3), (20_000.0, 100_000.0, 20_000.0))
        oracle = FakeOracle(
            (2, 3),
            {
                (2,): route((2,), (20_000.0, 40_000.0)),
                (2, 3): exact,
            },
            baseline_distance_m=200_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        req = request()
        context = generator.build_request_context(req)
        generator.build_s2_context(context, 0)
        lambda_context = generator.build_lambda_context(context, 0, 1)
        lambda_index = int(np.flatnonzero(lambda_context.mask)[0])
        action = HierarchicalAction(0, 1, lambda_index)

        plan = generator.materialize_plan(context, action)
        calls_after_materialization = list(oracle.route_calls)
        generator.validate_plan(req, plan)

        self.assertIs(plan.route, exact)
        self.assertEqual(plan.lambda1, float(lambda_context.bins[lambda_index]))
        self.assertEqual(oracle.route_calls, calls_after_materialization)
        self.assertEqual(plan.detour_ratio, exact.distance_m / 200_000.0 - 1.0)

    def test_materialize_indexes_selected_context_without_probing_other_routes(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {(2,): route((2,), (20_000.0, 40_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())
        s2_context = generator.build_s2_context(context, 0)
        self.assertTrue(s2_context.mask[generator.none_index])
        calls_before_materialization = list(oracle.route_calls)

        plan = generator.materialize_plan(
            context,
            HierarchicalAction(0, generator.none_index, None),
        )

        self.assertEqual(plan.s1, 2)
        self.assertEqual(oracle.route_calls, calls_before_materialization)

    def test_cached_empty_s1_context_raises_on_every_call(self) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (500_000.0, 100_000.0))},
            baseline_distance_m=1_000_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        with self.assertRaises(NoFeasibleChargingPlanError):
            generator.build_s1_context(context)
        with self.assertRaises(NoFeasibleChargingPlanError):
            generator.build_s1_context(context)

    def test_decision_arrays_cannot_be_mutated_to_enable_invalid_action(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {
                (2,): route((2,), (20_000.0, 800_000.0)),
                (2, 3): route((2, 3), (20_000.0, 100_000.0, 800_000.0)),
            },
            baseline_distance_m=1_000_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())
        s2_context = generator.build_s2_context(context, 0)
        lambda_context = generator.build_lambda_context(context, 0, 1)

        for array, index in (
            (s2_context.mask, 1),
            (s2_context.features, (1, 0)),
            (lambda_context.mask, 0),
            (lambda_context.features, 0),
            (lambda_context.bins, 0),
        ):
            with self.assertRaises(ValueError):
                array[index] = 1
            with self.assertRaises(ValueError):
                array.setflags(write=True)

        with self.assertRaises(InvalidHierarchicalActionError):
            generator.materialize_plan(context, HierarchicalAction(0, 1, 0))

    def test_equivalent_request_reuses_context_and_preserves_old_plan(self) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (20_000.0, 40_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        first_request = request()
        first_context = generator.build_request_context(first_request)
        generator.build_s2_context(first_context, 0)
        plan = generator.materialize_plan(
            first_context,
            HierarchicalAction(0, generator.none_index, None),
        )

        second_context = generator.build_request_context(request())

        self.assertIs(second_context, first_context)
        self.assertEqual(oracle.baseline_calls, 1)
        generator.validate_plan(first_request, plan)

    def test_exact_routes_are_cached_across_requests_with_same_od(self) -> None:
        station_ids = (2, 3, 4)
        exact_routes = {
            (s1,): route((s1,), (20_000.0, 40_000.0)) for s1 in station_ids
        }
        exact_routes.update(
            {
                (s1, s2): route((s1, s2), (20_000.0, 20_000.0, 20_000.0))
                for s1 in station_ids
                for s2 in station_ids
                if s1 != s2
            }
        )
        oracle = FakeOracle(station_ids, exact_routes)
        generator = FeasiblePlanGenerator(oracle)
        first = generator.build_request_context(request(vehicle_id=17))
        generator.build_s1_context(first)
        calls_after_first = list(oracle.route_calls)

        second = generator.build_request_context(request(vehicle_id=18))
        generator.build_s1_context(second)

        self.assertIsNot(second, first)
        self.assertEqual(len(calls_after_first), 9)
        self.assertEqual(oracle.route_calls, calls_after_first)

    def test_split_route_uses_all_pairs_lower_bound_before_exact_oracle_call(self) -> None:
        oracle = FakeOracle(
            (2, 3),
            {
                (2,): route((2,), (20_000.0, 40_000.0)),
                (2, 3): route((2, 3), (20_000.0, 200_000.0, 20_000.0)),
            },
        )
        oracle.node_to_index = {ORIGIN: 0, 2: 1, 3: 2, DESTINATION: 3}
        oracle.distances_m = (
            (0.0, 20_000.0, 220_000.0, 60_000.0),
            (float("inf"), 0.0, 200_000.0, 40_000.0),
            (float("inf"), float("inf"), 0.0, 20_000.0),
            (float("inf"), float("inf"), float("inf"), 0.0),
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())

        s2_context = generator.build_s2_context(context, 0)

        self.assertFalse(s2_context.mask[1])
        self.assertNotIn((2, 3), oracle.route_calls)

    def test_split_lower_bound_rejects_middle_leg_beyond_target_soc(self) -> None:
        exact_split = route((2, 3), (10_000.0, 80_000.0, 20_000.0))
        oracle = FakeOracle(
            (2, 3),
            {(2, 3): exact_split},
            baseline_distance_m=100_000.0,
        )
        oracle.node_to_index = {ORIGIN: 0, 2: 1, 3: 2, DESTINATION: 3}
        oracle.distances_m = (
            (0.0, 10_000.0, 90_000.0, 100_000.0),
            (float("inf"), 0.0, 80_000.0, 100_000.0),
            (float("inf"), float("inf"), 0.0, 20_000.0),
            (float("inf"), float("inf"), float("inf"), 0.0),
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(
            request(
                initial_soc=0.50,
                target_soc=0.50,
                soc_min=0.10,
                battery_capacity=100.0,
                rho_kwh_per_km=1.0,
            )
        )

        lambdas = generator.build_lambda_context(context, 0, 1)

        self.assertIsNone(lambdas)
        self.assertNotIn((2, 3), oracle.route_calls)

    def test_exact_lambda_mask_caps_first_stop_soc_at_target_soc(self) -> None:
        exact_split = route((2, 3), (10_000.0, 40_000.0, 20_000.0))
        oracle = FakeOracle(
            (2, 3),
            {(2, 3): exact_split},
            baseline_distance_m=100_000.0,
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(
            request(
                initial_soc=0.50,
                target_soc=0.50,
                soc_min=0.10,
                battery_capacity=100.0,
                rho_kwh_per_km=1.0,
            )
        )

        lambdas = generator.build_lambda_context(context, 0, 1)

        self.assertIsNotNone(lambdas)
        assert lambdas is not None
        target_index = int(np.flatnonzero(np.isclose(lambdas.bins, 0.50))[0])
        self.assertTrue(lambdas.mask[target_index])
        self.assertFalse(lambdas.mask[lambdas.bins > 0.50].any())

    def test_invalid_action_error_exposes_vehicle_od_and_indices(self) -> None:
        oracle = FakeOracle(
            (2,),
            {(2,): route((2,), (20_000.0, 40_000.0))},
        )
        generator = FeasiblePlanGenerator(oracle)
        context = generator.build_request_context(request())
        action = HierarchicalAction(s1_index=4, s2_index=1, lambda_index=12)

        with self.assertRaises(InvalidHierarchicalActionError) as caught:
            generator.materialize_plan(context, action)

        error = caught.exception
        self.assertEqual(error.vehicle_id, 17)
        self.assertEqual((error.origin, error.destination), (ORIGIN, DESTINATION))
        self.assertEqual(error.initial_soc, 0.50)
        self.assertEqual(error.target_soc, 0.80)
        self.assertEqual(error.s1_index, 4)
        self.assertEqual(error.s2_index, 1)
        self.assertEqual(error.lambda_index, 12)
        self.assertIn("4", str(error))
        self.assertIn("12", str(error))


if __name__ == "__main__":
    import unittest

    unittest.main()
