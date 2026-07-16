from __future__ import annotations

import random
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chargingpilot.evaluation import (
    EVALUATION_CSV_FIELDS,
    REQUEST_CSV_FIELDS,
    HierarchicalEvaluationMetadata,
    HierarchicalEvaluationRecord,
    EvaluationEpisodeSpec,
    StationVisit,
    aggregate_hierarchical_evaluation,
    bootstrap_day_confidence_interval,
    select_mandatory_service_shortest,
    select_minimum_wait_single,
    select_random_feasible,
    run_hierarchical_evaluation,
    write_hierarchical_evaluation_csv,
    write_hierarchical_evaluation_json,
    write_hierarchical_request_csv,
    write_hierarchical_request_json,
)


class _FakeGenerator:
    def __init__(
        self,
        candidates: dict[tuple[int, int], tuple[float, tuple[int, ...]]],
    ) -> None:
        self.station_ids = (10, 20, 30)
        self.none_index = len(self.station_ids)
        self._candidates = candidates

    def build_s1_context(self, context: object) -> object:
        del context
        mask = np.asarray(
            [any(key[0] == index for key in self._candidates) for index in range(3)],
            dtype=np.bool_,
        )
        return SimpleNamespace(mask=mask)

    def build_s2_context(self, context: object, s1_index: int) -> object:
        del context
        mask = np.zeros(4, dtype=np.bool_)
        routes: list[object | None] = [None] * 4
        for (candidate_s1, s2_index), (distance, _lambdas) in self._candidates.items():
            if candidate_s1 == s1_index:
                mask[s2_index] = True
                routes[s2_index] = SimpleNamespace(distance_m=distance)
        return SimpleNamespace(mask=mask, routes=tuple(routes))

    def build_lambda_context(
        self, context: object, s1_index: int, s2_index: int
    ) -> object | None:
        del context
        _distance, lambdas = self._candidates[(s1_index, s2_index)]
        mask = np.zeros(15, dtype=np.bool_)
        mask[list(lambdas)] = True
        return SimpleNamespace(mask=mask)


def _context(baseline_station_id: int = 20) -> object:
    return SimpleNamespace(
        baseline=SimpleNamespace(station_id=baseline_station_id),
        station_ids=(10, 20, 30),
    )


class BaselineSelectorTests(unittest.TestCase):
    def test_mandatory_service_uses_canonical_single_even_when_longer(self) -> None:
        generator = _FakeGenerator(
            {
                (0, 3): (10.0, ()),
                (1, 3): (90.0, ()),
            }
        )

        selection = select_mandatory_service_shortest(generator, _context())

        self.assertEqual((selection.action.s1_index, selection.action.s2_index), (1, 3))
        self.assertIsNone(selection.action.lambda_index)
        self.assertFalse(selection.fallback_used)

    def test_mandatory_service_fallback_uses_exact_deterministic_ties(self) -> None:
        generator = _FakeGenerator(
            {
                (2, 3): (50.0, ()),
                (0, 2): (50.0, (4, 2)),
                (0, 1): (50.0, (7,)),
            }
        )

        selection = select_mandatory_service_shortest(generator, _context())

        self.assertEqual(
            (
                selection.action.s1_index,
                selection.action.s2_index,
                selection.action.lambda_index,
            ),
            (2, 3, None),
        )
        self.assertTrue(selection.fallback_used)

    def test_minimum_wait_selects_legal_single_with_station_index_tie_break(self) -> None:
        generator = _FakeGenerator(
            {
                (0, 3): (100.0, ()),
                (1, 3): (90.0, ()),
                (2, 3): (80.0, ()),
            }
        )

        selection = select_minimum_wait_single(
            generator,
            _context(),
            estimated_wait_minutes={10: 5.0, 20: 3.0, 30: 3.0},
        )

        self.assertEqual(selection.action.s1_index, 1)
        self.assertFalse(selection.fallback_used)

    def test_minimum_wait_falls_back_to_shortest_split_and_smallest_lambda(self) -> None:
        generator = _FakeGenerator(
            {
                (1, 0): (60.0, (1,)),
                (0, 2): (40.0, (5, 3)),
            }
        )

        selection = select_minimum_wait_single(
            generator,
            _context(),
            estimated_wait_minutes={10: 5.0, 20: 3.0, 30: 3.0},
        )

        self.assertEqual(
            (
                selection.action.s1_index,
                selection.action.s2_index,
                selection.action.lambda_index,
            ),
            (0, 2, 3),
        )
        self.assertTrue(selection.fallback_used)

    def test_seeded_random_returns_only_a_feasible_action(self) -> None:
        generator = _FakeGenerator(
            {
                (0, 3): (10.0, ()),
                (1, 2): (20.0, (1, 4)),
            }
        )

        first = select_random_feasible(generator, _context(), rng=random.Random(7))
        second = select_random_feasible(generator, _context(), rng=random.Random(7))

        self.assertEqual(first, second)
        self.assertIn(
            (
                first.action.s1_index,
                first.action.s2_index,
                first.action.lambda_index,
            ),
            {(0, 3, None), (1, 2, 1), (1, 2, 4)},
        )
        self.assertFalse(first.fallback_used)


class DayBootstrapTests(unittest.TestCase):
    def test_seed_7_resamples_day_summaries_with_fixed_bounds(self) -> None:
        interval = bootstrap_day_confidence_interval(
            {"day-1": 1.0, "day-2": 2.0, "day-3": 9.0, "day-4": 10.0},
            seed=7,
            resamples=1000,
        )

        self.assertEqual(interval.unit, "day")
        self.assertEqual(interval.days, 4)
        self.assertAlmostEqual(interval.lower, 1.5)
        self.assertAlmostEqual(interval.upper, 9.5)

    def test_aggregate_bootstraps_four_metrics_from_day_summaries(self) -> None:
        values = (1.0, 2.0, 9.0, 10.0)
        records: list[HierarchicalEvaluationRecord] = []
        template = _evaluation_records()[0]
        for index, value in enumerate(values, start=1):
            visit = replace(
                template.visits[0],
                wait_minutes=value,
                grid_energy_kwh=2.0 * value,
                curtailment_kwh=3.0 * value,
            )
            records.append(
                replace(
                    template,
                    vehicle_id=index,
                    day=f"day-{index}",
                    visits=(visit,),
                    detour_ratio=value / 100.0,
                )
            )

        overall = aggregate_hierarchical_evaluation(
            records, bootstrap_seed=7, bootstrap_resamples=1000
        )[0]

        self.assertEqual(overall.bootstrap_unit, "day")
        self.assertEqual(overall.bootstrap_days, 4)
        self.assertEqual(overall.bootstrap_seed, 7)
        self.assertEqual(overall.bootstrap_resamples, 1000)
        self.assertAlmostEqual(overall.wait_minutes_mean_ci_lower, 1.5)
        self.assertAlmostEqual(overall.wait_minutes_mean_ci_upper, 9.5)
        self.assertAlmostEqual(overall.grid_energy_kwh_mean_ci_lower, 3.0)
        self.assertAlmostEqual(overall.grid_energy_kwh_mean_ci_upper, 19.0)
        self.assertAlmostEqual(overall.curtailment_kwh_mean_ci_lower, 4.5)
        self.assertAlmostEqual(overall.curtailment_kwh_mean_ci_upper, 28.5)
        self.assertAlmostEqual(overall.detour_ratio_mean_ci_lower, 0.015)
        self.assertAlmostEqual(overall.detour_ratio_mean_ci_upper, 0.095)


def _evaluation_records() -> list[HierarchicalEvaluationRecord]:
    shared = {
        "checkpoint": "checkpoint-0004.pt",
        "split": "test",
        "seed": 7,
        "detour_limit": 0.60,
        "policy": "mandatory_service_shortest",
    }
    return [
        HierarchicalEvaluationRecord(
            **shared,
            vehicle_id=1,
            day="day-1",
            load_class="low",
            hour=8,
            s1_station_id=10,
            s2_station_id=None,
            visits=(StationVisit(10, 1, 2, 0.0, 10.0, 1.0, 5.0),),
            detour_ratio=0.0,
            fallback_used=False,
            service_feasible=True,
            path_feasible=True,
            soc_feasible=True,
            detour_feasible=True,
            empty_mask=False,
        ),
        HierarchicalEvaluationRecord(
            **shared,
            vehicle_id=2,
            day="day-1",
            load_class="low",
            hour=8,
            s1_station_id=10,
            s2_station_id=20,
            visits=(
                StationVisit(10, 1, 3, 5.0, 8.0, 0.5, 4.0),
                StationVisit(20, 2, 5, 15.0, 12.0, 1.5, 6.0),
            ),
            detour_ratio=0.2,
            fallback_used=True,
            service_feasible=True,
            path_feasible=False,
            soc_feasible=True,
            detour_feasible=True,
            empty_mask=False,
        ),
        HierarchicalEvaluationRecord(
            **shared,
            vehicle_id=3,
            day="day-2",
            load_class="high",
            hour=9,
            s1_station_id=10,
            s2_station_id=None,
            visits=(StationVisit(10, 1, 4, 40.0, 30.0, 3.0, 15.0),),
            detour_ratio=0.4,
            fallback_used=False,
            service_feasible=False,
            path_feasible=True,
            soc_feasible=False,
            detour_feasible=False,
            empty_mask=True,
        ),
        HierarchicalEvaluationRecord(
            **shared,
            vehicle_id=4,
            day="day-2",
            load_class="high",
            hour=9,
            s1_station_id=10,
            s2_station_id=20,
            visits=(
                StationVisit(10, 1, 6, 30.0, 15.0, 1.0, 8.0),
                StationVisit(20, 2, 7, 50.0, 25.0, 3.0, 12.0),
            ),
            detour_ratio=0.6,
            fallback_used=True,
            service_feasible=True,
            path_feasible=True,
            soc_feasible=True,
            detour_feasible=True,
            empty_mask=False,
        ),
    ]


class MetricAggregationTests(unittest.TestCase):
    def test_aggregates_complete_metrics_in_all_required_strata(self) -> None:
        summaries = aggregate_hierarchical_evaluation(_evaluation_records())

        self.assertEqual(
            {summary.group_dimension for summary in summaries},
            {"overall", "day", "load_class", "hour", "station", "plan_type"},
        )
        overall = next(item for item in summaries if item.group_dimension == "overall")
        self.assertEqual(overall.request_count, 4)
        self.assertEqual(overall.wait_minutes_total, 140.0)
        self.assertEqual(overall.wait_minutes_mean, 35.0)
        self.assertEqual(overall.wait_minutes_p50, 30.0)
        self.assertAlmostEqual(overall.wait_minutes_p95, 74.0)
        self.assertAlmostEqual(overall.wait_minutes_p99, 78.8)
        self.assertEqual(overall.wait_minutes_max, 80.0)
        self.assertEqual(overall.wait_over_15_rate, 0.75)
        self.assertEqual(overall.wait_over_30_rate, 0.50)
        self.assertEqual(overall.wait_over_60_rate, 0.25)
        self.assertEqual(overall.max_station_queue, 7)
        self.assertEqual(overall.grid_energy_kwh_total, 100.0)
        self.assertEqual(overall.grid_energy_kwh_mean, 25.0)
        self.assertEqual(overall.curtailment_kwh_total, 10.0)
        self.assertEqual(overall.curtailment_kwh_mean, 2.5)
        self.assertAlmostEqual(overall.detour_ratio_mean, 0.3)
        self.assertAlmostEqual(overall.detour_ratio_p95, 0.57)
        self.assertAlmostEqual(overall.detour_ratio_max, 0.6)
        self.assertEqual(overall.additional_stop_rate, 0.5)
        self.assertEqual(overall.fallback_count, 2)
        self.assertEqual(overall.fallback_rate, 0.5)
        self.assertEqual(overall.service_feasible_rate, 0.75)
        self.assertEqual(overall.path_feasible_rate, 0.75)
        self.assertEqual(overall.soc_feasible_rate, 0.75)
        self.assertEqual(overall.detour_feasible_rate, 0.75)
        self.assertEqual(overall.hard_feasible_rate, 0.5)
        self.assertEqual(overall.empty_mask_count, 1)

        station_ten = next(
            item
            for item in summaries
            if (item.group_dimension, item.group_value) == ("station", "10")
        )
        split = next(
            item
            for item in summaries
            if (item.group_dimension, item.group_value) == ("plan_type", "split")
        )
        station_twenty = next(
            item
            for item in summaries
            if (item.group_dimension, item.group_value) == ("station", "20")
        )
        self.assertEqual(station_ten.request_count, 4)
        self.assertEqual(station_twenty.request_count, 2)
        self.assertEqual(station_twenty.wait_minutes_total, 65.0)
        self.assertEqual(station_twenty.grid_energy_kwh_total, 37.0)
        self.assertEqual(split.request_count, 2)

    def test_rejects_mixed_run_metadata(self) -> None:
        records = _evaluation_records()
        records[0] = HierarchicalEvaluationRecord(
            **{**records[0].__dict__, "seed": 8}
        )

        with self.assertRaisesRegex(ValueError, "metadata"):
            aggregate_hierarchical_evaluation(records)


class EvaluationOutputTests(unittest.TestCase):
    def test_csv_and_json_use_stable_fields_and_run_metadata(self) -> None:
        records = _evaluation_records()
        summaries = aggregate_hierarchical_evaluation(records)
        metadata = HierarchicalEvaluationMetadata.from_records(records)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = write_hierarchical_evaluation_csv(
                summaries, Path(temp_dir) / "metrics.csv", metadata=metadata
            )
            json_path = write_hierarchical_evaluation_json(
                summaries, Path(temp_dir) / "metrics.json", metadata=metadata
            )
            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                csv_rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), EVALUATION_CSV_FIELDS)
                self.assertEqual(len(EVALUATION_CSV_FIELDS), len(set(EVALUATION_CSV_FIELDS)))
            json_payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(csv_rows[0]["checkpoint"], "checkpoint-0004.pt")
        self.assertEqual(csv_rows[0]["split"], "test")
        self.assertEqual(csv_rows[0]["seed"], "7")
        self.assertEqual(csv_rows[0]["detour_limit"], "0.6")
        self.assertEqual(csv_rows[0]["fallback_count"], "2")
        self.assertEqual(json_payload["checkpoint"], "checkpoint-0004.pt")
        self.assertEqual(json_payload["split"], "test")
        self.assertEqual(json_payload["seed"], 7)
        self.assertEqual(json_payload["detour_limit"], 0.6)
        self.assertEqual(json_payload["fallback_count"], 2)
        self.assertEqual(json_payload["fallback_rate"], 0.5)
        self.assertEqual(json_payload["bootstrap_seed"], 7)
        self.assertEqual(json_payload["bootstrap_resamples"], 10000)
        self.assertEqual(json_payload["groups"][0]["group_dimension"], "overall")

    def test_request_csv_and_json_preserve_vehicle_and_both_visit_waits(self) -> None:
        records = _evaluation_records()
        metadata = HierarchicalEvaluationMetadata.from_records(records)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = write_hierarchical_request_csv(
                records, Path(temp_dir) / "requests.csv", metadata=metadata
            )
            json_path = write_hierarchical_request_json(
                records, Path(temp_dir) / "requests.json", metadata=metadata
            )
            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                csv_rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), REQUEST_CSV_FIELDS)
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        split_csv = csv_rows[1]
        self.assertEqual(split_csv["vehicle_id"], "2")
        self.assertEqual(split_csv["s1_station_id"], "10")
        self.assertEqual(split_csv["s2_station_id"], "20")
        self.assertEqual(split_csv["wait_minutes_total"], "20.0")
        visits = json.loads(split_csv["visits_json"])
        self.assertEqual([item["station_id"] for item in visits], [10, 20])
        self.assertEqual([item["wait_minutes"] for item in visits], [5.0, 15.0])
        split_json = payload["requests"][1]
        self.assertIsNone(payload["requests"][0]["s2_station_id"])
        self.assertEqual(split_json["wait_minutes_total"], 20.0)
        self.assertEqual(len(split_json["visits"]), 2)

    def test_legacy_module_explicitly_reexports_hierarchical_api(self) -> None:
        from chargingpilot.evaluation import (
            EvaluationEpisodeSpec as ReexportedEpisodeSpec,
            HierarchicalEvaluationRecord as ReexportedRecord,
            StationVisit as ReexportedVisit,
            aggregate_hierarchical_evaluation as reexported_aggregate,
            run_hierarchical_evaluation as reexported_runner,
            write_hierarchical_request_csv as reexported_request_csv,
        )

        self.assertIs(ReexportedRecord, HierarchicalEvaluationRecord)
        self.assertIs(ReexportedEpisodeSpec, EvaluationEpisodeSpec)
        self.assertIs(ReexportedVisit, StationVisit)
        self.assertIs(reexported_aggregate, aggregate_hierarchical_evaluation)
        self.assertIs(reexported_runner, run_hierarchical_evaluation)
        self.assertIs(reexported_request_csv, write_hierarchical_request_csv)


class _DeterministicPolicy:
    def __init__(self) -> None:
        self.deterministic_flags: list[bool] = []

    def sample_action(
        self, observation: object, s1_context: object, context_provider: object, *, deterministic: bool
    ) -> object:
        del observation, s1_context, context_provider
        self.deterministic_flags.append(bool(deterministic))
        from chargingpilot.routing.models import HierarchicalAction

        return SimpleNamespace(action=HierarchicalAction(0, 1, 12))


class HierarchicalEvaluationRunnerTests(unittest.TestCase):
    def test_runner_marks_executed_soc_shortfall_as_hard_infeasible(self) -> None:
        from tests.test_hierarchical_split_charging_env import (
            _make_environment,
            _policy_request,
        )

        def environment_factory() -> object:
            environment, _network = _make_environment((_policy_request(42),))
            original_step = environment.step

            def step_with_soc_shortfall(action):
                result = original_step(action)
                if result[2]:
                    record = environment.simulator.history_log._records[-1]
                    environment.simulator.history_log._records[-1] = replace(
                        record,
                        end_soc=float(record.target_soc) - 0.10,
                    )
                return result

            environment.step = step_with_soc_shortfall
            return environment

        reports = run_hierarchical_evaluation(
            (
                EvaluationEpisodeSpec(
                    day="2026-07-01",
                    load_class="medium",
                    environment_factory=environment_factory,
                ),
            ),
            policy=_DeterministicPolicy(),
            checkpoint="policy.pt",
            split="validation",
            seed=7,
            detour_limit=0.60,
        )

        for report in reports.values():
            self.assertFalse(report.records[0].soc_feasible)
            self.assertEqual(report.aggregates[0].soc_feasible_rate, 0.0)
            self.assertEqual(report.aggregates[0].hard_feasible_rate, 0.0)

    def test_runner_executes_policy_and_exact_three_baselines_from_current_context(self) -> None:
        from tests.test_hierarchical_split_charging_env import (
            _make_environment,
            _policy_request,
        )

        factory_calls: list[int] = []

        def environment_factory() -> object:
            factory_calls.append(len(factory_calls))
            environment, _network = _make_environment((_policy_request(41),))
            return environment

        policy = _DeterministicPolicy()
        reports = run_hierarchical_evaluation(
            (
                EvaluationEpisodeSpec(
                    day="2026-07-01",
                    load_class="medium",
                    environment_factory=environment_factory,
                ),
            ),
            policy=policy,
            checkpoint="policy.pt",
            split="validation",
            seed=7,
            detour_limit=0.60,
        )

        self.assertEqual(
            tuple(reports),
            (
                "ppo",
                "mandatory_service_shortest",
                "minimum_wait_single",
                "random_feasible",
            ),
        )
        self.assertEqual(len(factory_calls), 4)
        self.assertEqual(policy.deterministic_flags, [True])
        for label, report in reports.items():
            self.assertEqual(report.metadata.policy, label)
            self.assertEqual(report.metadata.seed, 7)
            self.assertEqual(report.metadata.checkpoint, "policy.pt")
            self.assertEqual(report.metadata.bootstrap_seed, 7)
            self.assertEqual(report.metadata.bootstrap_resamples, 10000)
            self.assertEqual(len(report.records), 1)
            record = report.records[0]
            self.assertEqual(record.policy, label)
            self.assertEqual(record.vehicle_id, 41)
            self.assertFalse(record.empty_mask)
            self.assertTrue(record.service_feasible)
            self.assertTrue(record.path_feasible)
            self.assertTrue(record.soc_feasible)
            self.assertTrue(record.detour_feasible)
            self.assertEqual(record.wait_minutes, sum(visit.wait_minutes for visit in record.visits))
            self.assertEqual(report.aggregates[0].bootstrap_seed, 7)

        ppo_record = reports["ppo"].records[0]
        self.assertEqual((ppo_record.s1_station_id, ppo_record.s2_station_id), (100, 101))
        self.assertEqual([visit.station_id for visit in ppo_record.visits], [100, 101])
        self.assertEqual(len(ppo_record.visits), 2)


if __name__ == "__main__":
    unittest.main()
