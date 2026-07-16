import unittest

from chargingpilot.simulator.models import ChargingSocRequest, StationSpec, VehicleSpec
from chargingpilot.simulator.simulator import SimulatorCore
from chargingpilot.simulator.station import StationRuntime


def make_vehicle_spec(
    *,
    battery_capacity: float = 60.0,
    initial_soc: float = 0.4,
    p_max_kw: float = 60.0,
) -> VehicleSpec:
    return VehicleSpec(
        battery_capacity=battery_capacity,
        initial_soc=initial_soc,
        soc_min=0.0,
        p_max_kw=p_max_kw,
        p_min_kw=30.0,
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
    arrival_soc: float = 0.4,
    target_soc: float = 0.9,
    vehicle_spec: VehicleSpec | None = None,
) -> ChargingSocRequest:
    return ChargingSocRequest(
        vehicle_id=vehicle_id,
        station_id=station_id,
        arrival_time=0.0,
        vehicle_spec=vehicle_spec or make_vehicle_spec(initial_soc=arrival_soc),
        arrival_soc=arrival_soc,
        target_soc=target_soc,
    )


class StationEnergyLedgerTests(unittest.TestCase):
    def test_renewable_supplies_vehicle_before_grid(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_plug_kw=60.0,
                p_grid_max_kw=80.0,
                renewable_power_trace=((0.0, 80.0),),
                eta=1.0,
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(make_request(1))
        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertEqual(state.active_power_kw, [60.0])
        self.assertAlmostEqual(state.renewable_used_kw, 60.0)
        self.assertAlmostEqual(state.ess_discharge_kw, 0.0)
        self.assertAlmostEqual(state.grid_used_kw, 0.0)
        self.assertAlmostEqual(state.power_available_kw, 160.0)

    def test_ess_supplements_renewable_before_grid(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_plug_kw=60.0,
                p_grid_max_kw=80.0,
                renewable_power_trace=((0.0, 20.0),),
                ess_capacity_kwh=20.0,
                ess_initial_kwh=10.0,
                ess_discharge_efficiency=1.0,
                p_ess_discharge_max_kw=80.0,
                eta=1.0,
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(make_request(1))
        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertEqual(state.active_power_kw, [60.0])
        self.assertAlmostEqual(state.renewable_used_kw, 20.0)
        self.assertAlmostEqual(state.ess_discharge_kw, 40.0)
        self.assertAlmostEqual(state.grid_used_kw, 0.0)
        self.assertAlmostEqual(state.ess_energy_kwh, 10.0 - (40.0 / 60.0))

    def test_grid_is_capped_after_renewable_and_ess_are_exhausted(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_plug_kw=60.0,
                p_grid_max_kw=20.0,
                renewable_power_trace=((0.0, 10.0),),
                ess_capacity_kwh=1.0,
                ess_initial_kwh=0.25,
                ess_discharge_efficiency=1.0,
                p_ess_discharge_max_kw=60.0,
                eta=1.0,
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(make_request(1))
        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertEqual(state.active_power_kw, [30.0])
        self.assertAlmostEqual(state.renewable_used_kw, 10.0)
        self.assertAlmostEqual(state.ess_discharge_kw, 15.0)
        self.assertAlmostEqual(state.grid_used_kw, 20.0)
        self.assertAlmostEqual(state.ess_energy_kwh, 0.0)
        self.assertAlmostEqual(state.power_available_kw, 30.0)

    def test_surplus_renewable_charges_ess_before_curtailment(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_grid_max_kw=0.0,
                renewable_power_trace=((0.0, 100.0),),
                ess_capacity_kwh=1.0,
                ess_initial_kwh=0.9,
                ess_charge_efficiency=1.0,
                p_ess_charge_max_kw=30.0,
            ),
            timestep_minutes=1.0,
        )

        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertAlmostEqual(state.renewable_used_kw, 0.0)
        self.assertAlmostEqual(state.ess_charge_kw, 6.0)
        self.assertAlmostEqual(state.renewable_curtailed_kw, 94.0)
        self.assertAlmostEqual(state.grid_used_kw, 0.0)
        self.assertAlmostEqual(state.ess_energy_kwh, 1.0)

    def test_station_spec_does_not_accept_converter_power_limit(self) -> None:
        with self.assertRaises(TypeError):
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_conv_max_kw=50.0,
            )

    def test_station_power_uses_full_grid_renewable_and_ess_supply(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=2,
                p_plug_kw=100.0,
                p_grid_max_kw=100.0,
                renewable_power_trace=((0.0, 100.0),),
                eta=1.0,
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(
            make_request(1, vehicle_spec=make_vehicle_spec(p_max_kw=100.0))
        )
        station.enqueue_soc_request(
            make_request(2, vehicle_spec=make_vehicle_spec(p_max_kw=100.0))
        )
        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertEqual(state.active_power_kw, [100.0, 100.0])
        self.assertAlmostEqual(state.renewable_used_kw, 100.0)
        self.assertAlmostEqual(state.grid_used_kw, 100.0)
        self.assertAlmostEqual(state.power_available_kw, 200.0)

    def test_external_ess_power_trace_is_clipped_by_energy_bounds(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_plug_kw=60.0,
                p_grid_max_kw=60.0,
                renewable_power_trace=((0.0, 0.0),),
                ess_capacity_kwh=1.0,
                ess_initial_kwh=0.25,
                ess_discharge_efficiency=1.0,
                p_ess_discharge_max_kw=60.0,
                ess_power_trace=((0.0, 60.0),),
                eta=1.0,
            ),
            timestep_minutes=1.0,
        )

        station.enqueue_soc_request(make_request(1))
        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertEqual(state.active_power_kw, [60.0])
        self.assertAlmostEqual(state.ess_discharge_kw, 15.0)
        self.assertAlmostEqual(state.grid_used_kw, 45.0)
        self.assertAlmostEqual(state.ess_energy_kwh, 0.0)

    def test_external_negative_ess_power_trace_charges_from_surplus_renewable(self) -> None:
        station = StationRuntime(
            StationSpec(
                station_id=0,
                charge_capacity=1,
                p_grid_max_kw=0.0,
                renewable_power_trace=((0.0, 100.0),),
                ess_capacity_kwh=1.0,
                ess_initial_kwh=0.0,
                ess_charge_efficiency=1.0,
                p_ess_charge_max_kw=100.0,
                ess_power_trace=((0.0, -30.0),),
            ),
            timestep_minutes=1.0,
        )

        state = station.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertAlmostEqual(state.ess_charge_kw, 30.0)
        self.assertAlmostEqual(state.renewable_curtailed_kw, 70.0)
        self.assertAlmostEqual(state.ess_energy_kwh, 0.5)

    def test_snapshot_preserves_energy_ledger_and_ess_state(self) -> None:
        spec = StationSpec(
            station_id=0,
            charge_capacity=1,
            p_plug_kw=60.0,
            p_grid_max_kw=80.0,
            renewable_power_trace=((0.0, 20.0),),
            ess_capacity_kwh=20.0,
            ess_initial_kwh=10.0,
            ess_discharge_efficiency=1.0,
            p_ess_discharge_max_kw=80.0,
            eta=1.0,
        )
        station = StationRuntime(spec, timestep_minutes=1.0)

        station.enqueue_soc_request(make_request(1))
        station.to_state(query_time=1.0, queue_waiting_time=[])
        snapshot = station.snapshot()
        restored = StationRuntime(spec, timestep_minutes=1.0)
        restored.restore(snapshot)
        restored_state = restored.to_state(query_time=1.0, queue_waiting_time=[])

        self.assertAlmostEqual(snapshot.renewable_used_kw, 20.0)
        self.assertAlmostEqual(snapshot.ess_discharge_kw, 40.0)
        self.assertAlmostEqual(snapshot.grid_used_kw, 0.0)
        self.assertAlmostEqual(restored_state.ess_energy_kwh, 10.0 - (40.0 / 60.0))
        self.assertAlmostEqual(restored_state.ess_discharge_kw, 40.0)

    def test_simulator_metrics_include_cumulative_station_energy_use(self) -> None:
        core = SimulatorCore(
            station_specs=[
                StationSpec(
                    station_id=0,
                    charge_capacity=1,
                    p_plug_kw=60.0,
                    p_grid_max_kw=80.0,
                    renewable_power_trace=((0.0, 20.0),),
                    ess_capacity_kwh=20.0,
                    ess_initial_kwh=10.0,
                    ess_discharge_efficiency=1.0,
                    p_ess_discharge_max_kw=80.0,
                    eta=1.0,
                )
            ],
            timestep_minutes=1.0,
        )

        core.enqueue_soc_arrival(make_request(1))
        core.advance_to(1.0)
        metrics = core.get_metrics()
        state = core.get_state()

        self.assertAlmostEqual(metrics.renewable_used_kwh[0], 20.0 / 60.0)
        self.assertAlmostEqual(metrics.ess_discharged_kwh[0], 40.0 / 60.0)
        self.assertAlmostEqual(metrics.grid_used_kwh[0], 0.0)
        self.assertAlmostEqual(metrics.ess_charged_kwh[0], 0.0)
        self.assertAlmostEqual(metrics.renewable_curtailed_kwh[0], 0.0)
        self.assertAlmostEqual(
            state["metrics"]["renewable_used_kwh"][0],
            20.0 / 60.0,
        )


if __name__ == "__main__":
    unittest.main()
