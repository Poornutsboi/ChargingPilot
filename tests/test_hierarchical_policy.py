from __future__ import annotations

import unittest

import numpy as np
import torch

from chargingpilot.environment.models import VehicleRequest
from chargingpilot.environment.structured_observation import StructuredObservation
from chargingpilot.routing.models import (
    HierarchicalAction,
    LambdaDecisionContext,
    RequestFeasibilityContext,
    S1DecisionContext,
    S2DecisionContext,
    ServiceBaseline,
)
from chargingpilot.simulator.models import VehicleSpec
from chargingpilot.trainer.hierarchical_policy import HierarchicalActorCritic


STATION_IDS = tuple(range(100, 172))
STATION_NODE_INDICES = tuple(range(10, 82))


def _request(vehicle_id: int = 7) -> VehicleRequest:
    return VehicleRequest(
        vehicle_id=vehicle_id,
        decision_time=0.0,
        vehicle_spec=VehicleSpec(
            battery_capacity=80.0,
            initial_soc=0.4,
            soc_min=0.1,
            p_max_kw=120.0,
            p_min_kw=0.0,
            rho_kwh_per_km=0.15,
            origin=1,
            destination=2,
            departure_time=0.0,
            path_nodes=(1, 2),
            path_edges=(),
        ),
        target_soc=0.8,
    )


def _request_context(vehicle_id: int = 7) -> RequestFeasibilityContext:
    return RequestFeasibilityContext(
        request=_request(vehicle_id),
        baseline=ServiceBaseline(100, 1_000.0, (1, 100, 2)),
        station_ids=STATION_IDS,
        single_routes={},
        split_routes={},
        arrival_soc_s1={},
    )


def _observation(*, request_shift: float = 0.0, station_shift: float = 0.0) -> StructuredObservation:
    request = np.linspace(0.0, 1.0, 16, dtype=np.float32) + np.float32(request_shift)
    stations = np.linspace(0.0, 1.0, 72 * 33, dtype=np.float32).reshape(72, 33)
    stations = stations + np.float32(station_shift)
    return StructuredObservation(1, 2, request, stations.astype(np.float32))


def _contexts(
    *,
    s1_index: int = 3,
    s2_index: int = 5,
    lambda_index: int = 4,
    none: bool = False,
    request_context: RequestFeasibilityContext | None = None,
) -> tuple[S1DecisionContext, S2DecisionContext, LambdaDecisionContext | None]:
    context = request_context or _request_context()
    s1_mask = np.zeros(72, dtype=np.bool_)
    s1_mask[s1_index] = True
    s2_mask = np.zeros(73, dtype=np.bool_)
    selected_s2 = 72 if none else s2_index
    s2_mask[selected_s2] = True
    pair_features = np.linspace(0.0, 1.0, 73 * 6, dtype=np.float32).reshape(73, 6)
    s1_context = S1DecisionContext(context, s1_mask)
    s2_context = S2DecisionContext(
        context,
        s1_index,
        s2_mask,
        pair_features,
        (None,) * 73,
    )
    if none:
        return s1_context, s2_context, None
    lambda_mask = np.zeros(15, dtype=np.bool_)
    lambda_mask[lambda_index] = True
    lambda_context = LambdaDecisionContext(
        context,
        s1_index,
        s2_index,
        lambda_mask,
        np.linspace(0.1, 0.5, 5, dtype=np.float32),
        np.linspace(0.3, 1.0, 15, dtype=np.float32),
    )
    return s1_context, s2_context, lambda_context


class _Provider:
    def __init__(
        self,
        s2_context: S2DecisionContext,
        lambda_context: LambdaDecisionContext | None,
    ) -> None:
        self.s2_context = s2_context
        self.lambda_context = lambda_context

    def build_s2_context(self, request_context, s1_index):
        if request_context is not self.s2_context.request_context:
            raise AssertionError("wrong request context")
        if s1_index != self.s2_context.s1_index:
            raise AssertionError("wrong s1 index")
        return self.s2_context

    def build_lambda_context(self, request_context, s1_index, s2_index):
        if request_context is not self.s2_context.request_context:
            raise AssertionError("wrong request context")
        if (s1_index, s2_index) != (
            self.s2_context.s1_index,
            s2_index,
        ):
            raise AssertionError("wrong indices")
        return self.lambda_context


class HierarchicalPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)

    def test_policy_constructs_with_production_output_sizes(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )

        self.assertEqual(policy.s1_count, 72)
        self.assertEqual(policy.s2_count, 73)
        self.assertEqual(policy.lambda_count, 15)

    def test_station_graph_nodes_distinguish_identical_continuous_rows(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100,
            station_node_indices=STATION_NODE_INDICES,
        )
        observation = StructuredObservation(
            1,
            2,
            np.zeros(16, dtype=np.float32),
            np.zeros((72, 33), dtype=np.float32),
        )

        encoded = policy.encode(observation)
        logits = policy.s1_distribution(
            encoded, np.ones(72, dtype=np.bool_)
        ).logits

        self.assertFalse(torch.equal(encoded.stations[0, 0], encoded.stations[0, 1]))
        self.assertNotEqual(
            float(logits[0, 0].detach()), float(logits[0, 1].detach())
        )

    def test_sampling_never_selects_masked_logits_and_is_deterministic(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )
        s1_context, s2_context, lambda_context = _contexts()
        provider = _Provider(s2_context, lambda_context)

        for deterministic in (False, True):
            for _ in range(10):
                sample = policy.sample_action(
                    _observation(), s1_context, provider, deterministic=deterministic
                )
                self.assertEqual(sample.action, HierarchicalAction(3, 5, 4))
                self.assertTrue(torch.isfinite(sample.log_prob))
                self.assertTrue(torch.isfinite(sample.value))

    def test_none_action_omits_lambda_term(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )
        s1_context, s2_context, lambda_context = _contexts(none=True)
        action = HierarchicalAction(3, 72, None)

        evaluation = policy.evaluate_action(
            _observation(), action, s1_context, s2_context, lambda_context
        )
        encoded = policy.encode(_observation())
        s1_dist = policy.s1_distribution(encoded, s1_context.mask)
        s2_dist = policy.s2_distribution(
            encoded, torch.tensor([3]), s2_context.features, s2_context.mask
        )
        expected = s1_dist.log_prob(torch.tensor([3])) + s2_dist.log_prob(
            torch.tensor([72])
        )

        self.assertTrue(torch.allclose(evaluation.log_prob.reshape(1), expected))
        self.assertEqual(float(evaluation.lambda_entropy), 0.0)

    def test_split_joint_log_probability_is_sum_of_three_heads(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )
        s1_context, s2_context, lambda_context = _contexts()
        action = HierarchicalAction(3, 5, 4)

        evaluation = policy.evaluate_action(
            _observation(), action, s1_context, s2_context, lambda_context
        )
        encoded = policy.encode(_observation())
        expected = (
            policy.s1_distribution(encoded, s1_context.mask).log_prob(torch.tensor([3]))
            + policy.s2_distribution(
                encoded, torch.tensor([3]), s2_context.features, s2_context.mask
            ).log_prob(torch.tensor([5]))
            + policy.lambda_distribution(
                encoded,
                torch.tensor([3]),
                torch.tensor([5]),
                lambda_context.features,
                lambda_context.mask,
            ).log_prob(torch.tensor([4]))
        )

        self.assertTrue(torch.allclose(evaluation.log_prob.reshape(1), expected))

    def test_head_shapes_and_conditioning_inputs(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )
        observation = _observation()
        encoded = policy.encode(observation)
        s1_context, s2_context, lambda_context = _contexts()

        s1_logits = policy.s1_distribution(encoded, s1_context.mask).logits
        s2_logits = policy.s2_distribution(
            encoded, torch.tensor([3]), s2_context.features, np.ones(73, dtype=np.bool_)
        ).logits
        lambda_logits = policy.lambda_distribution(
            encoded,
            torch.tensor([3]),
            torch.tensor([5]),
            lambda_context.features,
            np.ones(15, dtype=np.bool_),
        ).logits
        self.assertEqual(tuple(s1_logits.shape), (1, 72))
        self.assertEqual(tuple(s2_logits.shape), (1, 73))
        self.assertEqual(tuple(lambda_logits.shape), (1, 15))

        changed_s1 = policy.s2_distribution(
            encoded, torch.tensor([60]), s2_context.features, np.ones(73, dtype=np.bool_)
        ).logits
        changed_pairs = policy.s2_distribution(
            encoded,
            torch.tensor([3]),
            s2_context.features + np.float32(0.25),
            np.ones(73, dtype=np.bool_),
        ).logits
        self.assertFalse(torch.allclose(s2_logits, changed_s1))
        self.assertFalse(torch.allclose(s2_logits, changed_pairs))

        changed_station = policy.lambda_distribution(
            encoded,
            torch.tensor([4]),
            torch.tensor([6]),
            lambda_context.features,
            np.ones(15, dtype=np.bool_),
        ).logits
        changed_context = policy.lambda_distribution(
            encoded,
            torch.tensor([3]),
            torch.tensor([5]),
            lambda_context.features + np.float32(0.25),
            np.ones(15, dtype=np.bool_),
        ).logits
        self.assertFalse(torch.allclose(lambda_logits, changed_station))
        self.assertFalse(torch.allclose(lambda_logits, changed_context))

    def test_value_uses_request_and_pooled_station_context(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )

        base = policy.encode(_observation()).value
        changed_request = policy.encode(_observation(request_shift=0.5)).value
        changed_stations = policy.encode(_observation(station_shift=0.5)).value

        self.assertFalse(torch.allclose(base, changed_request))
        self.assertFalse(torch.allclose(base, changed_stations))

    def test_none_embedding_and_every_active_head_receive_gradients(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )
        none_contexts = list(_contexts(none=True))
        split_contexts = list(_contexts())
        for contexts, alternative_s2 in ((none_contexts, 5), (split_contexts, 6)):
            s1_mask = contexts[0].mask.copy()
            s1_mask[4] = True
            s2_mask = contexts[1].mask.copy()
            s2_mask[alternative_s2] = True
            contexts[0] = S1DecisionContext(contexts[0].request_context, s1_mask)
            contexts[1] = S2DecisionContext(
                contexts[0].request_context,
                3,
                s2_mask,
                contexts[1].features,
                contexts[1].routes,
            )
        lambda_mask = split_contexts[2].mask.copy()
        lambda_mask[5] = True
        split_contexts[2] = LambdaDecisionContext(
            split_contexts[0].request_context,
            3,
            5,
            lambda_mask,
            split_contexts[2].features,
            split_contexts[2].bins,
        )
        none_eval = policy.evaluate_action(
            _observation(),
            HierarchicalAction(3, 72, None),
            *none_contexts,
        )
        split_eval = policy.evaluate_action(
            _observation(station_shift=0.1),
            HierarchicalAction(3, 5, 4),
            *split_contexts,
        )

        loss = -(none_eval.log_prob + split_eval.log_prob) + (
            none_eval.value + split_eval.value
        ).square().sum()
        loss.backward()

        self.assertGreater(float(policy.none_station_embedding.grad.abs().sum()), 0.0)
        for module in (policy.s1_head, policy.s2_head, policy.lambda_head, policy.value_head):
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_rejects_empty_masks_context_mismatch_and_bad_observation_dtype(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        )
        encoded = policy.encode(_observation())
        with self.assertRaisesRegex(ValueError, "at least one feasible"):
            policy.s1_distribution(encoded, np.zeros(72, dtype=np.bool_))

        s1_context, s2_context, lambda_context = _contexts()
        other_context = _request_context(vehicle_id=99)
        mismatched_s2 = S2DecisionContext(
            other_context,
            3,
            s2_context.mask,
            s2_context.features,
            s2_context.routes,
        )
        with self.assertRaisesRegex(ValueError, "request context"):
            policy.evaluate_action(
                _observation(),
                HierarchicalAction(3, 5, 4),
                s1_context,
                mismatched_s2,
                lambda_context,
            )

        bad = object.__new__(StructuredObservation)
        object.__setattr__(bad, "origin_index", 1)
        object.__setattr__(bad, "destination_index", 2)
        object.__setattr__(bad, "request", np.zeros(16, dtype=np.float64))
        object.__setattr__(bad, "stations", np.zeros((72, 33), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "float32"):
            policy.encode(bad)

    def test_batched_evaluation_preserves_batch_shape_device_and_dtype(self) -> None:
        policy = HierarchicalActorCritic(
            node_count=100, station_node_indices=STATION_NODE_INDICES
        ).to(dtype=torch.float64)
        single = _contexts(none=True)
        split = _contexts()

        evaluation = policy.evaluate_action(
            [_observation(), _observation(station_shift=0.1)],
            [HierarchicalAction(3, 72, None), HierarchicalAction(3, 5, 4)],
            [single[0], split[0]],
            [single[1], split[1]],
            [single[2], split[2]],
        )

        for tensor in (
            evaluation.log_prob,
            evaluation.value,
            evaluation.s1_entropy,
            evaluation.s2_entropy,
            evaluation.lambda_entropy,
        ):
            self.assertEqual(tuple(tensor.shape), (2,))
            self.assertEqual(tensor.dtype, torch.float64)
            self.assertEqual(tensor.device, policy.node_embedding.weight.device)
        self.assertEqual(float(evaluation.lambda_entropy[0].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
