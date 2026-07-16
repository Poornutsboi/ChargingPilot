import unittest

import chargingpilot.simulator as simulator
from chargingpilot.simulator import SimulatorCore
from chargingpilot.simulator.models import ChargingSocRequest, StationSpec, VehicleSpec
from chargingpilot.simulator.station import StationRuntime


def make_vehicle_spec(
    *,
    battery_capacity: float = 60.0,
    initial_soc: float = 0.4,
    p_max_kw: float = 120.0,
    p_min_kw: float = 30.0,
) -> VehicleSpec:
    return VehicleSpec(
        battery_capacity=battery_capacity,
        initial_soc=initial_soc,
        soc_min=0.0,
        p_max_kw=p_max_kw,
        p_min_kw=p_min_kw,
        rho_kwh_per_km=0.18,
        origin=0,
        destination=1,
        departure_time=0.0,
        path_nodes=(0, 1),
        path_edges=("0->1",),
    )


def make_request(
    vehicle_id: int,
    *,
    station_id: int = 0,
    arrival_time: float = 0.0,
    arrival_soc: float = 0.4,
    target_soc: float = 0.505,
    vehicle_spec: VehicleSpec | None = None,
) -> ChargingSocRequest:
    return ChargingSocRequest(
        vehicle_id=vehicle_id,
        station_id=station_id,
        arrival_time=arrival_time,
        vehicle_spec=vehicle_spec or make_vehicle_spec(initial_soc=arrival_soc),
        arrival_soc=arrival_soc,
        target_soc=target_soc,
    )


class SimulatorFixedTimestepTests(unittest.TestCase):
    def test_main_package_exports_station_runtime(self) -> None:
        self.assertIs(simulator.SimulatorCore, SimulatorCore)
        self.assertIs(simulator.StationRuntime, StationRuntime)

    def test_event_driven_public_methods_are_removed(self) -> None:
        core = SimulatorCore(
            station_specs=[
                StationSpec(
                    station_id=0,
                    charge_capacity=1,
                    p_plug_kw=60.0,
                    p_max_kw=60.0,
                    eta=1.0,
                )
            ],
            timestep_minutes=1.0,
        )

        self.assertFalse(hasattr(core, "next_charging_event_time"))
        self.assertFalse(hasattr(core, "advance_next_charging_event"))

    def test_submit_soc_arrival_quantizes_completion_to_one_minute_tick(self) -> None:
        core = SimulatorCore(
            station_specs=[
                StationSpec(
                    station_id=0,
                    charge_capacity=1,
                    p_plug_kw=60.0,
                    p_max_kw=60.0,
                    eta=1.0,
                )
            ],
            timestep_minutes=1.0,
        )

        assignment = core.submit_soc_arrival(make_request(1, target_soc=0.505))

        self.assertAlmostEqual(assignment.start_time, 0.0)
        self.assertAlmostEqual(assignment.end_time, 7.0)
        self.assertAlmostEqual(assignment.wait_time, 0.0)
        self.assertAlmostEqual(assignment.start_soc, 0.4)
        self.assertAlmostEqual(assignment.end_soc, 0.505)
        self.assertAlmostEqual(assignment.energy_delivered_kwh, 6.3)

    def test_queue_starts_after_discrete_completion_tick(self) -> None:
        core = SimulatorCore(
            station_specs=[
                StationSpec(
                    station_id=0,
                    charge_capacity=1,
                    p_plug_kw=60.0,
                    p_max_kw=60.0,
                    eta=1.0,
                )
            ],
            timestep_minutes=1.0,
        )

        core.enqueue_soc_arrival(make_request(1, target_soc=0.505))
        core.enqueue_soc_arrival(make_request(2, target_soc=0.45))

        first_tick = core.advance_to(7.0)
        second_tick = core.advance_to(10.0)

        self.assertEqual([assignment.vehicle_id for assignment in first_tick], [1])
        self.assertEqual([assignment.vehicle_id for assignment in second_tick], [2])
        self.assertAlmostEqual(second_tick[0].start_time, 7.0)
        self.assertAlmostEqual(second_tick[0].wait_time, 7.0)
        self.assertAlmostEqual(second_tick[0].end_time, 10.0)

    def test_active_sessions_share_station_power_each_tick(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=2,
                p_plug_kw=60.0,
                p_max_kw=60.0,
                eta=1.0,
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(make_request(1, target_soc=0.9))
        station.enqueue_soc_request(make_request(2, target_soc=0.9))
        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertEqual(state.active_vehicle_ids, [1, 2])
        self.assertEqual(state.active_power_kw, [30.0, 30.0])
        self.assertAlmostEqual(state.active_soc[0], 0.4083333333)
        self.assertAlmostEqual(state.active_soc[1], 0.4083333333)

    def test_power_trace_changes_are_applied_on_fixed_ticks(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=2,
                p_plug_kw=100.0,
                p_max_kw=120.0,
                eta=1.0,
                power_trace=((0.0, 120.0), (5.0, 60.0)),
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(make_request(1, target_soc=0.9))
        station.enqueue_soc_request(make_request(2, target_soc=0.9))
        before = station.to_state(query_time=4.0, queue_waiting_time=[])
        after = station.to_state(query_time=6.0, queue_waiting_time=[])

        self.assertEqual(before.active_power_kw, [60.0, 60.0])
        self.assertEqual(after.active_power_kw, [30.0, 30.0])

    def test_scheduled_soc_arrivals_are_independent_from_fixed_ticks(self) -> None:
        core = SimulatorCore(
            station_specs=[StationSpec(station_id=0, charge_capacity=1)],
            timestep_minutes=1.0,
        )
        later = make_request(1, arrival_time=10.0, target_soc=0.5)
        earlier = make_request(2, arrival_time=5.0, target_soc=0.5)

        core.schedule_soc_arrival(later)
        core.schedule_soc_arrival(earlier)

        self.assertAlmostEqual(core.next_scheduled_soc_arrival_time(), 5.0)
        self.assertEqual(core.pop_due_scheduled_soc_arrivals(4.0), [])
        self.assertEqual(core.pop_due_scheduled_soc_arrivals(5.0), [earlier])
        self.assertAlmostEqual(core.next_scheduled_soc_arrival_time(), 10.0)


if __name__ == "__main__":
    unittest.main()
