from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import InvalidHierarchicalActionError, NoFeasibleChargingPlanError
from .models import (
    ChargingPlan,
    HierarchicalAction,
    LambdaDecisionContext,
    RequestFeasibilityContext,
    RouteResult,
    S1DecisionContext,
    S2DecisionContext,
)

if TYPE_CHECKING:
    from chargingpilot.environment.models import VehicleRequest


class FeasiblePlanGenerator:
    DETOUR_LIMIT = 0.60
    SOC_EPSILON = 1e-9

    def __init__(
        self,
        oracle: Any,
        station_ids: tuple[int, ...] | None = None,
        *,
        detour_limit: float = DETOUR_LIMIT,
    ) -> None:
        selected_ids = oracle.station_ids if station_ids is None else station_ids
        self.oracle = oracle
        self.station_ids = tuple(int(station_id) for station_id in selected_ids)
        if not self.station_ids or any(
            left >= right
            for left, right in zip(self.station_ids, self.station_ids[1:])
        ):
            raise ValueError("station_ids must be strictly ascending and unique")
        self.none_index = len(self.station_ids)
        self.detour_limit = float(detour_limit)
        if self.detour_limit <= 0.0:
            raise ValueError("detour_limit must be positive")
        self._bin_values = np.round(np.arange(0.30, 1.0001, 0.05), 2)
        self.bins = self._bin_values.astype(np.float32)
        self._s1_contexts: dict[int, S1DecisionContext] = {}
        self._s2_contexts: dict[tuple[int, int], S2DecisionContext] = {}
        self._lambda_contexts: dict[
            tuple[int, int, int], LambdaDecisionContext | None
        ] = {}
        self._failed_single_routes: set[tuple[int, int]] = set()
        self._failed_split_routes: set[tuple[int, int, int]] = set()
        self._route_arrival_soc: dict[tuple[int, int], float] = {}
        self._route_geometry_cache: dict[
            tuple[int, int, tuple[int, ...]], RouteResult
        ] = {}
        self._failed_route_geometries: set[tuple[int, int, tuple[int, ...]]] = set()
        self._request_contexts: dict[tuple[object, ...], RequestFeasibilityContext] = {}

    def build_request_context(
        self, request: VehicleRequest
    ) -> RequestFeasibilityContext:
        request_key = self._request_key(request)
        existing = self._request_contexts.get(request_key)
        if existing is not None:
            return existing
        spec = request.vehicle_spec
        baseline = self.oracle.service_baseline(spec.origin, spec.destination)
        context = RequestFeasibilityContext(
            request=request,
            baseline=baseline,
            station_ids=self.station_ids,
            single_routes={},
            split_routes={},
            arrival_soc_s1={},
        )
        self._request_contexts[request_key] = context
        return context

    def register_context(
        self, context: RequestFeasibilityContext
    ) -> RequestFeasibilityContext:
        if not isinstance(context, RequestFeasibilityContext):
            raise TypeError("context must be a RequestFeasibilityContext")
        if tuple(context.station_ids) != self.station_ids:
            raise ValueError("request context station mapping does not match generator")
        spec = context.request.vehicle_spec
        expected_baseline = self.oracle.service_baseline(
            int(spec.origin), int(spec.destination)
        )
        if context.baseline != expected_baseline:
            raise ValueError("request context baseline does not match generator oracle")

        station_count = len(self.station_ids)
        for raw_index, route in context.single_routes.items():
            index = int(raw_index)
            if index != raw_index or not 0 <= index < station_count:
                raise ValueError("request context contains an invalid single-route index")
            if not self._route_matches(route, (self.station_ids[index],)):
                raise ValueError("request context contains a mismatched single route")
        for raw_pair, route in context.split_routes.items():
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise ValueError("request context contains an invalid split-route index")
            s1_index, s2_index = (int(value) for value in raw_pair)
            if (
                raw_pair != (s1_index, s2_index)
                or not 0 <= s1_index < station_count
                or not 0 <= s2_index < station_count
                or s1_index == s2_index
            ):
                raise ValueError("request context contains an invalid split-route index")
            if not self._route_matches(
                route, (self.station_ids[s1_index], self.station_ids[s2_index])
            ):
                raise ValueError("request context contains a mismatched split route")
        if any(
            int(raw_index) != raw_index
            or not 0 <= int(raw_index) < station_count
            for raw_index in context.arrival_soc_s1
        ):
            raise ValueError("request context contains an invalid arrival-SOC index")

        request_key = self._request_key(context.request)
        existing = self._request_contexts.get(request_key)
        if existing is not None and existing is not context:
            raise ValueError("request key already has a different canonical context")
        self._request_contexts[request_key] = context

        origin = int(spec.origin)
        destination = int(spec.destination)
        for route in (*context.single_routes.values(), *context.split_routes.values()):
            geometry_key = (origin, destination, tuple(route.required_station_ids))
            self._route_geometry_cache[geometry_key] = route
        for s1_index, arrival_soc in context.arrival_soc_s1.items():
            route = context.single_routes.get(int(s1_index))
            if route is not None:
                self._route_arrival_soc[(id(context), id(route))] = float(arrival_soc)
        return context

    def build_s1_context(
        self, context: RequestFeasibilityContext
    ) -> S1DecisionContext:
        self._require_owned_context(context)
        key = id(context)
        existing = self._s1_contexts.get(key)
        if existing is not None:
            if not existing.mask.any():
                raise NoFeasibleChargingPlanError(context.request, existing.mask)
            return existing

        mask = np.zeros(len(self.station_ids), dtype=np.bool_)
        for s1_index in range(len(self.station_ids)):
            child = self.build_s2_context(context, s1_index)
            mask[s1_index] = child.mask.any()
        result = S1DecisionContext(context, mask)
        self._s1_contexts[key] = result
        if not mask.any():
            raise NoFeasibleChargingPlanError(context.request, mask)
        return result

    def find_first_feasible_action(
        self, context: RequestFeasibilityContext
    ) -> HierarchicalAction | None:
        self._require_owned_context(context)
        baseline_index = self._station_index(context.baseline.station_id)
        s1_indices = (
            *((baseline_index,) if baseline_index >= 0 else ()),
            *(index for index in range(len(self.station_ids)) if index != baseline_index),
        )
        for s1_index in s1_indices:
            single = self._single_route(context, s1_index)
            if single is not None and self._single_is_valid(
                context, s1_index, single
            ):
                return HierarchicalAction(s1_index, self.none_index, None)
            for s2_index in range(len(self.station_ids)):
                if s2_index == s1_index:
                    continue
                lambdas = self.build_lambda_context(context, s1_index, s2_index)
                if lambdas is None:
                    continue
                feasible_lambdas = np.flatnonzero(lambdas.mask)
                if feasible_lambdas.size:
                    return HierarchicalAction(
                        s1_index, s2_index, int(feasible_lambdas[0])
                    )
        return None

    def build_s2_context(
        self,
        context: RequestFeasibilityContext,
        s1_index: int,
    ) -> S2DecisionContext:
        self._require_owned_context(context)
        s1_index = int(s1_index)
        if not 0 <= s1_index < len(self.station_ids):
            raise IndexError(f"s1_index {s1_index} is out of range")
        cache_key = (id(context), s1_index)
        existing = self._s2_contexts.get(cache_key)
        if existing is not None:
            return existing

        size = len(self.station_ids) + 1
        mask = np.zeros(size, dtype=np.bool_)
        features = np.zeros((size, 6), dtype=np.float32)
        routes: list[RouteResult | None] = [None] * size
        s1 = self.station_ids[s1_index]

        single = self._single_route(context, s1_index)
        if single is not None:
            routes[self.none_index] = single
            features[self.none_index] = self._s2_features(
                context, single, leg_distance_m=0.0, is_none=True
            )
            mask[self.none_index] = self._single_is_valid(context, s1_index, single)

        for s2_index, s2 in enumerate(self.station_ids):
            if s2_index == s1_index:
                continue
            split = self._split_route(context, s1_index, s2_index)
            if split is None:
                continue
            routes[s2_index] = split
            leg_distance_m = split.legs[1].distance_m
            features[s2_index] = self._s2_features(
                context, split, leg_distance_m=leg_distance_m, is_none=False
            )
            lambda_context = self.build_lambda_context(
                context, s1_index, s2_index
            )
            mask[s2_index] = bool(
                lambda_context is not None and lambda_context.mask.any()
            )

        result = S2DecisionContext(
            request_context=context,
            s1_index=s1_index,
            mask=mask,
            features=features,
            routes=tuple(routes),
        )
        self._s2_contexts[cache_key] = result
        return result

    def build_lambda_context(
        self,
        context: RequestFeasibilityContext,
        s1_index: int,
        s2_index: int,
    ) -> LambdaDecisionContext | None:
        self._require_owned_context(context)
        s1_index = int(s1_index)
        s2_index = int(s2_index)
        if s2_index == self.none_index:
            return None
        if not 0 <= s1_index < len(self.station_ids):
            raise IndexError(f"s1_index {s1_index} is out of range")
        if not 0 <= s2_index < len(self.station_ids):
            raise IndexError(f"s2_index {s2_index} is out of range")
        cache_key = (id(context), s1_index, s2_index)
        if cache_key in self._lambda_contexts:
            return self._lambda_contexts[cache_key]
        if s1_index == s2_index:
            self._lambda_contexts[cache_key] = None
            return None

        route = self._split_route(context, s1_index, s2_index)
        if route is None or len(route.legs) != 3:
            self._lambda_contexts[cache_key] = None
            return None

        arrival_soc = self._arrival_soc(context, s1_index, route)
        energy_12 = self._energy_fraction(context.request, route.legs[1].distance_m)
        energy_2d = self._energy_fraction(context.request, route.legs[2].distance_m)
        target_soc = float(context.request.target_soc)
        soc_min = float(context.request.vehicle_spec.soc_min)
        detour_ratio = self._detour_ratio(context, route)
        features = np.asarray(
            [
                arrival_soc,
                energy_12,
                target_soc,
                self._distance_ratio(context, route),
                detour_ratio / self.detour_limit,
            ],
            dtype=np.float32,
        )
        mask = np.zeros(15, dtype=np.bool_)
        if (
            self._route_matches(route, (self.station_ids[s1_index], self.station_ids[s2_index]))
            and self._within_detour(context, route)
            and arrival_soc >= soc_min - self.SOC_EPSILON
            and target_soc - energy_2d >= soc_min - self.SOC_EPSILON
        ):
            for index, lambda1 in enumerate(self._bin_values):
                arrival_soc_s2 = float(lambda1) - energy_12
                mask[index] = (
                    float(lambda1) > arrival_soc + self.SOC_EPSILON
                    and float(lambda1) <= target_soc + self.SOC_EPSILON
                    and arrival_soc_s2 >= soc_min - self.SOC_EPSILON
                    and target_soc - arrival_soc_s2 > self.SOC_EPSILON
                )
        result = LambdaDecisionContext(
            request_context=context,
            s1_index=s1_index,
            s2_index=s2_index,
            mask=mask,
            features=features,
            bins=self.bins.copy(),
        )
        self._lambda_contexts[cache_key] = result
        return result

    def materialize_plan(
        self,
        context: RequestFeasibilityContext,
        action: HierarchicalAction,
    ) -> ChargingPlan:
        self._require_owned_context(context)
        try:
            s1_index = int(action.s1_index)
            s2_index = int(action.s2_index)
            if not 0 <= s1_index < len(self.station_ids):
                raise ValueError("s1 index is out of range")
            if not 0 <= s2_index <= self.none_index:
                raise ValueError("s2 index is out of range")
            s1_context = self._s1_contexts.get(id(context))
            if s1_context is not None and not s1_context.mask[s1_index]:
                raise ValueError("s1 is masked")
            s2_context = self._s2_contexts.get((id(context), s1_index))
            if s2_context is not None:
                if not s2_context.mask[s2_index]:
                    raise ValueError("s2 is masked")
                route = s2_context.routes[s2_index]
                if route is None:
                    raise ValueError("selected route is unavailable")
            elif s2_index == self.none_index:
                route = self._single_route(context, s1_index)
                if route is None or not self._single_is_valid(
                    context, s1_index, route
                ):
                    raise ValueError("single-stop route is infeasible")
            else:
                route = self._split_route(context, s1_index, s2_index)
                if route is None:
                    raise ValueError("selected split route is unavailable")

            if s2_index == self.none_index:
                if action.lambda_index is not None:
                    raise ValueError("single-stop action must omit lambda index")
                s2 = None
                lambda1 = float(context.request.target_soc)
            else:
                if action.lambda_index is None:
                    raise ValueError("split action requires lambda index")
                lambda_index = int(action.lambda_index)
                if not 0 <= lambda_index < len(self._bin_values):
                    raise ValueError("lambda index is out of range")
                lambda_context = self._lambda_contexts.get(
                    (id(context), s1_index, s2_index)
                )
                if lambda_context is None:
                    lambda_context = self.build_lambda_context(
                        context, s1_index, s2_index
                    )
                if lambda_context is None or not lambda_context.mask[lambda_index]:
                    raise ValueError("lambda is masked")
                s2 = self.station_ids[s2_index]
                lambda1 = float(self._bin_values[lambda_index])
        except (IndexError, NoFeasibleChargingPlanError, ValueError) as exc:
            raise InvalidHierarchicalActionError(
                context.request, action, str(exc)
            ) from exc

        plan = ChargingPlan(
            vehicle_id=int(context.request.vehicle_id),
            s1=self.station_ids[s1_index],
            s2=s2,
            lambda1=lambda1,
            baseline=context.baseline,
            route=route,
            detour_ratio=self._detour_ratio(context, route),
        )
        self.validate_plan(context.request, plan)
        return plan

    def validate_plan(self, request: VehicleRequest, plan: ChargingPlan) -> None:
        context = self._request_contexts.get(self._request_key(request))
        s1_index = self._station_index(plan.s1)
        s2_index = self.none_index if plan.s2 is None else self._station_index(plan.s2)
        lambda_index = self._lambda_index(plan.lambda1) if plan.s2 is not None else None
        action = HierarchicalAction(s1_index, s2_index, lambda_index)

        def invalid(reason: str) -> None:
            raise InvalidHierarchicalActionError(request, action, reason)

        if int(plan.vehicle_id) != int(request.vehicle_id):
            invalid("vehicle id does not match request")
        if context is None:
            invalid("request feasibility context has not been built")
        if s1_index < 0 or s2_index < 0:
            invalid("plan contains an unknown station id")
        if plan.baseline != context.baseline:
            invalid("plan does not contain the canonical baseline")
        if s2_index == self.none_index:
            expected_route = context.single_routes.get(s1_index)
            if expected_route is not plan.route:
                invalid("plan does not contain the stored exact single route")
            if abs(float(plan.lambda1) - float(request.target_soc)) > self.SOC_EPSILON:
                invalid("single-stop lambda must equal target SOC")
            if not self._single_is_valid(context, s1_index, plan.route):
                invalid("single-stop SOC or detour constraint is invalid")
        else:
            expected_route = context.split_routes.get((s1_index, s2_index))
            if expected_route is not plan.route:
                invalid("plan does not contain the stored exact split route")
            if lambda_index is None or lambda_index < 0:
                invalid("split lambda does not match a configured bin")
            lambda_context = self.build_lambda_context(context, s1_index, s2_index)
            if lambda_context is None or not lambda_context.mask[lambda_index]:
                invalid("split lambda is masked")

        expected_detour = self._detour_ratio(context, plan.route)
        ratio_epsilon = self._distance_epsilon(context) / max(
            float(context.baseline.distance_m), self._distance_epsilon(context)
        )
        if abs(float(plan.detour_ratio) - expected_detour) > ratio_epsilon:
            invalid("detour ratio does not match the stored route")

    def _single_route(
        self, context: RequestFeasibilityContext, s1_index: int
    ) -> RouteResult | None:
        if s1_index in context.single_routes:
            return context.single_routes[s1_index]
        key = (id(context), s1_index)
        if key in self._failed_single_routes:
            return None
        station_id = self.station_ids[s1_index]
        result = self._exact_route(context, (station_id,))
        if result is None:
            self._failed_single_routes.add(key)
            return None
        context.single_routes[s1_index] = result
        self._arrival_soc(context, s1_index, result)
        return result

    def _split_route(
        self,
        context: RequestFeasibilityContext,
        s1_index: int,
        s2_index: int,
    ) -> RouteResult | None:
        pair = (s1_index, s2_index)
        if pair in context.split_routes:
            return context.split_routes[pair]
        key = (id(context), s1_index, s2_index)
        if key in self._failed_split_routes:
            return None
        required = (self.station_ids[s1_index], self.station_ids[s2_index])
        if not self._split_lower_bound_can_be_feasible(
            context, s1_index, s2_index
        ):
            self._failed_split_routes.add(key)
            return None
        result = self._exact_route(context, required)
        if result is None:
            self._failed_split_routes.add(key)
            return None
        context.split_routes[pair] = result
        self._arrival_soc(context, s1_index, result)
        return result

    def _exact_route(
        self,
        context: RequestFeasibilityContext,
        required: tuple[int, ...],
    ) -> RouteResult | None:
        spec = context.request.vehicle_spec
        key = (int(spec.origin), int(spec.destination), required)
        existing = self._route_geometry_cache.get(key)
        if existing is not None:
            return existing
        if key in self._failed_route_geometries:
            return None
        try:
            result = self.oracle.route_via(spec.origin, spec.destination, required)
        except ValueError:
            self._failed_route_geometries.add(key)
            return None
        self._route_geometry_cache[key] = result
        return result

    def _split_lower_bound_can_be_feasible(
        self,
        context: RequestFeasibilityContext,
        s1_index: int,
        s2_index: int,
    ) -> bool:
        spec = context.request.vehicle_spec
        s1 = self.station_ids[s1_index]
        s2 = self.station_ids[s2_index]
        distances = (
            self._oracle_distance(spec.origin, s1),
            self._oracle_distance(s1, s2),
            self._oracle_distance(s2, spec.destination),
        )
        if any(distance is None for distance in distances):
            return True
        lower_bound = tuple(float(distance) for distance in distances)
        if not all(math.isfinite(distance) for distance in lower_bound):
            return False
        if sum(lower_bound) > (
            (1.0 + self.detour_limit) * float(context.baseline.distance_m)
            + self._distance_epsilon(context)
        ):
            return False
        soc_min = float(spec.soc_min)
        target_soc = float(context.request.target_soc)
        return bool(
            float(spec.initial_soc)
            - self._energy_fraction(context.request, lower_bound[0])
            >= soc_min - self.SOC_EPSILON
            and target_soc
            - self._energy_fraction(context.request, lower_bound[1])
            >= soc_min - self.SOC_EPSILON
            and target_soc
            - self._energy_fraction(context.request, lower_bound[2])
            >= soc_min - self.SOC_EPSILON
        )

    def _oracle_distance(self, source: int, target: int) -> float | None:
        indices = getattr(self.oracle, "node_to_index", None)
        distances = getattr(self.oracle, "distances_m", None)
        if indices is None or distances is None:
            return None
        try:
            return float(distances[indices[int(source)]][indices[int(target)]])
        except (KeyError, IndexError, TypeError):
            return None

    def _arrival_soc(
        self,
        context: RequestFeasibilityContext,
        s1_index: int,
        route: RouteResult,
    ) -> float:
        cache_key = (id(context), id(route))
        existing = self._route_arrival_soc.get(cache_key)
        if existing is not None:
            return existing
        if not route.legs:
            return -np.inf
        arrival = float(context.request.vehicle_spec.initial_soc) - self._energy_fraction(
            context.request, route.legs[0].distance_m
        )
        self._route_arrival_soc[cache_key] = arrival
        if len(route.required_station_ids) == 1:
            context.arrival_soc_s1[s1_index] = arrival
        return arrival

    def _single_is_valid(
        self,
        context: RequestFeasibilityContext,
        s1_index: int,
        route: RouteResult,
    ) -> bool:
        if len(route.legs) != 2 or not self._route_matches(
            route, (self.station_ids[s1_index],)
        ):
            return False
        request = context.request
        arrival = self._arrival_soc(context, s1_index, route)
        target = float(request.target_soc)
        soc_min = float(request.vehicle_spec.soc_min)
        final_energy = self._energy_fraction(request, route.legs[1].distance_m)
        return bool(
            self._within_detour(context, route)
            and arrival >= soc_min - self.SOC_EPSILON
            and target - arrival > self.SOC_EPSILON
            and target - final_energy >= soc_min - self.SOC_EPSILON
        )

    def _s2_features(
        self,
        context: RequestFeasibilityContext,
        route: RouteResult,
        *,
        leg_distance_m: float,
        is_none: bool,
    ) -> np.ndarray:
        detour = self._detour_ratio(context, route)
        return np.asarray(
            [
                1.0 if is_none else 0.0,
                float(leg_distance_m) / float(context.baseline.distance_m),
                self._energy_fraction(context.request, leg_distance_m),
                self._distance_ratio(context, route),
                detour / self.detour_limit,
                0.0 if is_none else 1.0,
            ],
            dtype=np.float32,
        )

    def _within_detour(
        self, context: RequestFeasibilityContext, route: RouteResult
    ) -> bool:
        bound = (1.0 + self.detour_limit) * float(context.baseline.distance_m)
        return float(route.distance_m) <= bound + self._distance_epsilon(context)

    def _distance_epsilon(self, context: RequestFeasibilityContext) -> float:
        return max(1e-6, 1e-9 * float(context.baseline.distance_m))

    @staticmethod
    def _route_matches(route: RouteResult, required: tuple[int, ...]) -> bool:
        return route.required_station_ids == required

    @staticmethod
    def _distance_ratio(
        context: RequestFeasibilityContext, route: RouteResult
    ) -> float:
        baseline = float(context.baseline.distance_m)
        return float(route.distance_m) / baseline

    def _detour_ratio(
        self, context: RequestFeasibilityContext, route: RouteResult
    ) -> float:
        return self._distance_ratio(context, route) - 1.0

    @staticmethod
    def _energy_fraction(request: VehicleRequest, distance_m: float) -> float:
        spec = request.vehicle_spec
        return (
            float(spec.rho_kwh_per_km)
            * (float(distance_m) / 1_000.0)
            / float(spec.battery_capacity)
        )

    def _station_index(self, station_id: int) -> int:
        try:
            return self.station_ids.index(int(station_id))
        except ValueError:
            return -1

    def _lambda_index(self, value: float) -> int:
        for index, bin_value in enumerate(self._bin_values):
            if abs(float(value) - float(bin_value)) <= self.SOC_EPSILON:
                return index
        return -1

    def _require_owned_context(self, context: RequestFeasibilityContext) -> None:
        if context.station_ids != self.station_ids:
            raise ValueError("request context station mapping does not match generator")
        if self._request_contexts.get(self._request_key(context.request)) is not context:
            raise ValueError("request context is not the canonical generator context")

    @staticmethod
    def _request_key(request: VehicleRequest) -> tuple[object, ...]:
        spec = request.vehicle_spec
        return (
            int(request.vehicle_id),
            int(spec.origin),
            int(spec.destination),
            float(spec.initial_soc),
            float(spec.soc_min),
            float(spec.battery_capacity),
            float(spec.rho_kwh_per_km),
            float(request.target_soc),
        )
