import importlib.util
import math
import sys
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "chargingpilot" / "roadmap" / "generate_charging_requests.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_charging_requests", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RoadMapChargingRequestTests(unittest.TestCase):
    def test_first_service_distance_uses_path_order(self):
        module = load_module()
        edge_lengths = {
            (1, 2): 40_000.0,
            (2, 3): 30_000.0,
            (3, 4): 20_000.0,
        }

        service_id, distance_m = module.first_service_on_path([1, 2, 3, 4], {3, 4}, edge_lengths)

        self.assertEqual(service_id, 3)
        self.assertEqual(distance_m, 70_000.0)

    def test_start_soc_covers_energy_to_nearest_service(self):
        module = load_module()

        start_soc = module.feasible_start_soc(
            battery_kwh=75.0,
            rho_kwh_per_km=0.2,
            distance_to_service_km=120.0,
            safety_soc=0.04,
            rng=module.random.Random(7),
        )

        self.assertGreaterEqual(start_soc, 120.0 * 0.2 / 75.0 + 0.04)
        self.assertLessEqual(start_soc, 0.92)

    def test_row_contains_date_time_and_feasible_initial_energy(self):
        module = load_module()
        request = module.build_request_row(
            vehicle_index=1,
            day_index=0,
            minute_of_day=75,
            origin=10,
            destination=20,
            path_nodes=[10, 11, 12, 20],
            path_length_km=260.0,
            nearest_service_id=12,
            distance_to_service_km=85.0,
            battery_kwh=90.0,
            rho_kwh_per_km=0.18,
            start_soc=0.25,
            target_soc=0.9,
        )

        self.assertEqual(request["date"], "01-01")
        self.assertEqual(request["datetime"], "01-01 01:15")
        self.assertEqual(request["arrival_time"], "01:15")
        self.assertEqual(request["o_i"], 10)
        self.assertEqual(request["d_i"], 20)
        self.assertEqual(request["nearest_service_id"], 12)
        self.assertTrue(math.isclose(request["energy_to_nearest_service_kwh"], 15.3))
        self.assertGreaterEqual(
            request["start_soc"] * request["B_i"],
            request["energy_to_nearest_service_kwh"],
        )


if __name__ == "__main__":
    unittest.main()
