from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from chargingpilot.simulator.incoming import IncomingLoadTracker


def plan(vehicle_id: int, s1: int, s2: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(vehicle_id=vehicle_id, s1=s1, s2=s2)


class IncomingLoadTrackerTests(TestCase):
    def test_constructor_rejects_non_integer_station_ids(self) -> None:
        for station_ids in ("25", (True, 5), (2.0, 5), ("2", 5)):
            with self.subTest(station_ids=station_ids):
                with self.assertRaises(ValueError):
                    IncomingLoadTracker(station_ids)

    def test_rejects_duplicate_records_and_negative_energy(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(1, 2), 10.0, 4.0)

        with self.assertRaisesRegex(ValueError, "already exists"):
            tracker.add_plan(plan(1, 2), 12.0, 3.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            tracker.add_plan(plan(2, 2), 10.0, -0.1)
        with self.assertRaisesRegex(ValueError, "station"):
            tracker.add_plan(plan(3, 9), 10.0, 1.0)

    def test_adds_first_and_provisional_second_records_atomically(self) -> None:
        tracker = IncomingLoadTracker((2, 5))

        tracker.add_plan(
            plan(7, 2, 5),
            first_arrival_time=10.0,
            first_kwh=8.0,
            provisional_second_arrival_time=40.0,
            second_kwh=12.0,
        )

        summary = tracker.summarize(now=0.0)
        np.testing.assert_array_equal(
            summary.counts,
            np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            summary.kwh,
            np.array([[8, 0, 0], [0, 0, 12]], dtype=np.float32),
        )
        records = tracker.snapshot()["records"]
        self.assertFalse(records[0]["provisional"])
        self.assertTrue(records[1]["provisional"])

    def test_places_exact_eta_boundaries_in_fixed_windows(self) -> None:
        tracker = IncomingLoadTracker((2,))
        for vehicle_id, eta in enumerate((14.999, 15.0, 30.0, 60.0, 61.0), 1):
            tracker.add_plan(plan(vehicle_id, 2), eta, float(vehicle_id))

        summary = tracker.summarize(now=0.0)

        self.assertEqual(summary.station_ids, (2,))
        np.testing.assert_array_equal(
            summary.counts, np.array([[1, 1, 2]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            summary.kwh, np.array([[1, 2, 7]], dtype=np.float32)
        )
        np.testing.assert_allclose(summary.eta_min, np.array([14.999], np.float32))
        np.testing.assert_allclose(
            summary.eta_mean,
            np.array([(14.999 + 15.0 + 30.0 + 60.0 + 61.0) / 5], np.float32),
        )
        self.assertEqual(summary.counts.dtype, np.float32)
        with self.assertRaises(ValueError):
            summary.counts[0, 0] = 9
        with self.assertRaises(ValueError):
            summary.counts.setflags(write=True)

    def test_arrival_removes_only_the_addressed_leg(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(7, 2, 5), 10.0, 8.0, 40.0, 12.0)

        tracker.mark_arrived(7, 1)

        summary = tracker.summarize(now=0.0)
        np.testing.assert_array_equal(summary.counts[:, 0], np.array([0, 0]))
        np.testing.assert_array_equal(summary.counts[:, 2], np.array([0, 1]))

    def test_actual_second_leg_update_replaces_provisional_values(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(7, 2, 5), 10.0, 8.0, 40.0, 12.0)

        tracker.update_second_leg(7, expected_arrival_time=20.0, expected_kwh=9.0)

        second = tracker.snapshot()["records"][1]
        self.assertEqual(second["expected_arrival_time"], 20.0)
        self.assertEqual(second["expected_kwh"], 9.0)
        self.assertFalse(second["provisional"])

    def test_actual_second_leg_update_creates_missing_provisional_record(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(7, 2, 5), 10.0, 8.0)
        tracker.mark_arrived(7, 1)

        tracker.update_second_leg(7, expected_arrival_time=20.0, expected_kwh=9.0)

        records = tracker.snapshot()["records"]
        self.assertEqual(
            records,
            [
                {
                    "vehicle_id": 7,
                    "leg_index": 2,
                    "station_id": 5,
                    "expected_arrival_time": 20.0,
                    "expected_kwh": 9.0,
                    "provisional": False,
                }
            ],
        )

    def test_snapshot_restore_is_an_exact_validated_roundtrip(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(7, 2, 5), 10.0, 8.0, 40.0, 12.0)
        snapshot = tracker.snapshot()

        restored = IncomingLoadTracker.restore(snapshot)

        self.assertEqual(restored.snapshot(), snapshot)
        invalid = tracker.snapshot()
        invalid["records"][0]["expected_kwh"] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            IncomingLoadTracker.restore(invalid)

    def test_restore_rejects_malformed_snapshot_containers_and_keys(self) -> None:
        snapshot = IncomingLoadTracker((2, 5)).snapshot()
        malformed = [[]]

        missing = deepcopy(snapshot)
        del missing["records"]
        malformed.append(missing)
        extra = deepcopy(snapshot)
        extra["unexpected"] = []
        malformed.append(extra)
        station_string = deepcopy(snapshot)
        station_string["station_ids"] = "25"
        malformed.append(station_string)
        station_tuple = deepcopy(snapshot)
        station_tuple["station_ids"] = (2, 5)
        malformed.append(station_tuple)
        records_dict = deepcopy(snapshot)
        records_dict["records"] = {}
        malformed.append(records_dict)
        record_entry_list = deepcopy(snapshot)
        record_entry_list["records"] = [[]]
        malformed.append(record_entry_list)
        second_stations_dict = deepcopy(snapshot)
        second_stations_dict["second_station_ids"] = {}
        malformed.append(second_stations_dict)
        second_station_entry_list = deepcopy(snapshot)
        second_station_entry_list["second_station_ids"] = [[]]
        malformed.append(second_station_entry_list)

        for invalid in malformed:
            with self.subTest(snapshot=invalid):
                with self.assertRaises(ValueError):
                    IncomingLoadTracker.restore(invalid)

    def test_restore_rejects_lossy_integer_scalars(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(7, 2, 5), 10.0, 8.0, 40.0, 12.0)
        base = tracker.snapshot()
        mutations = (
            ("station_ids", 0, None, True),
            ("station_ids", 0, None, 2.0),
            ("station_ids", 0, None, "2"),
            ("records", 0, "vehicle_id", True),
            ("records", 0, "leg_index", 1.0),
            ("records", 0, "station_id", "2"),
            ("second_station_ids", 0, "vehicle_id", 7.0),
            ("second_station_ids", 0, "station_id", True),
        )

        for collection, index, field, value in mutations:
            invalid = deepcopy(base)
            if field is None:
                invalid[collection][index] = value
            else:
                invalid[collection][index][field] = value
            with self.subTest(collection=collection, field=field, value=value):
                with self.assertRaises(ValueError):
                    IncomingLoadTracker.restore(invalid)

    def test_restore_rejects_invalid_float_and_boolean_scalars(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(7, 2, 5), 10.0, 8.0, 40.0, 12.0)
        base = tracker.snapshot()
        mutations = (
            ("expected_arrival_time", "10.0"),
            ("expected_arrival_time", True),
            ("expected_arrival_time", float("inf")),
            ("expected_kwh", "8.0"),
            ("expected_kwh", False),
            ("expected_kwh", float("nan")),
            ("provisional", 0),
        )

        for field, value in mutations:
            invalid = deepcopy(base)
            invalid["records"][0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    IncomingLoadTracker.restore(invalid)

    def test_restore_rejects_first_leg_marked_provisional(self) -> None:
        tracker = IncomingLoadTracker((2,))
        tracker.add_plan(plan(7, 2), 10.0, 8.0)
        snapshot = tracker.snapshot()
        snapshot["records"][0]["provisional"] = True

        with self.assertRaisesRegex(ValueError, "first-leg"):
            IncomingLoadTracker.restore(snapshot)

    def test_past_due_record_must_be_removed_before_summary(self) -> None:
        tracker = IncomingLoadTracker((2,))
        tracker.add_plan(plan(1, 2), 9.0, 1.0)

        with self.assertRaisesRegex(ValueError, "past due"):
            tracker.summarize(now=10.0)

    def test_past_due_provisional_second_leg_is_excluded_from_summary(self) -> None:
        tracker = IncomingLoadTracker((2, 5))
        tracker.add_plan(plan(1, 2, 5), 10.0, 8.0, 9.0, 12.0)

        summary = tracker.summarize(now=10.0)

        np.testing.assert_array_equal(
            summary.counts,
            np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
