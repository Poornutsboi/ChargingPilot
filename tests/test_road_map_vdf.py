import importlib.util
import math
import sys
import tempfile
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "chargingpilot" / "roadmap" / "vdf.py"


def load_module():
    spec = importlib.util.spec_from_file_location("road_map_vdf", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RoadMapVDFTests(unittest.TestCase):
    def test_modified_vdf_travel_time_is_monotonic_around_capacity(self):
        module = load_module()

        free_flow = module.modified_vdf_travel_time_min(
            length_m=1000.0,
            free_flow_speed_kmph=60.0,
            flow_veh_per_hour=0.0,
            capacity_veh_per_hour=1000.0,
            m=2.5,
        )
        at_capacity = module.modified_vdf_travel_time_min(
            length_m=1000.0,
            free_flow_speed_kmph=60.0,
            flow_veh_per_hour=1000.0,
            capacity_veh_per_hour=1000.0,
            m=2.5,
        )
        congested = module.modified_vdf_travel_time_min(
            length_m=1000.0,
            free_flow_speed_kmph=60.0,
            flow_veh_per_hour=1500.0,
            capacity_veh_per_hour=1000.0,
            m=2.5,
        )

        self.assertTrue(math.isclose(free_flow, 1.0))
        self.assertGreaterEqual(at_capacity, free_flow)
        self.assertGreater(congested, at_capacity)

    def test_modified_vdf_clamps_extreme_volume_capacity_ratio(self):
        module = load_module()

        travel_time = module.modified_vdf_travel_time_min(
            length_m=1000.0,
            free_flow_speed_kmph=60.0,
            flow_veh_per_hour=5000.0,
            capacity_veh_per_hour=1000.0,
            m=2.5,
        )

        self.assertTrue(math.isfinite(travel_time))
        self.assertGreater(travel_time, 1.0)

    def test_load_hourly_flows_accepts_existing_wide_csv_format(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "link_24h_flows.csv"
            hour_columns = ",".join(f"flow_h{hour:02d}" for hour in range(24))
            hour_values = ",".join(str(100 + hour) for hour in range(24))
            path.write_text(
                f"link_id,from_id,to_id,{hour_columns}\n"
                f"7,1,2,{hour_values}\n",
                encoding="utf-8",
            )

            flows = module.load_hourly_flows(path)

        self.assertEqual(flows[7][0], 100.0)
        self.assertEqual(flows[7][8], 108.0)
        self.assertEqual(flows[7][23], 123.0)


if __name__ == "__main__":
    unittest.main()
