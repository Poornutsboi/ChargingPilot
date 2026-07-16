import importlib.util
import math
import sys
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "chargingpilot" / "roadmap" / "generate_link_24h_flow.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_link_24h_flow", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RoadMapTrafficFlowTests(unittest.TestCase):
    def test_region_score_is_higher_near_shanghai_than_remote_area(self):
        module = load_module()

        shanghai = module.classify_region(121.47, 31.23)
        remote = module.classify_region(119.45, 29.55)

        self.assertGreater(shanghai.score, remote.score)
        self.assertEqual(shanghai.nearest_city, "Shanghai")

    def test_hourly_profile_has_24_hours_and_commute_peaks(self):
        module = load_module()

        profile = module.hourly_profile()

        self.assertEqual(len(profile), 24)
        self.assertGreater(profile[8], profile[3])
        self.assertGreater(profile[18], profile[3])
        self.assertTrue(math.isclose(sum(profile) / len(profile), 1.0, rel_tol=1e-9))

    def test_connectivity_propagates_upstream_intensity(self):
        module = load_module()
        links = [
            {"link_id": 1, "from_id": 1, "to_id": 2, "initial_score": 5.0},
            {"link_id": 2, "from_id": 2, "to_id": 3, "initial_score": 1.0},
            {"link_id": 3, "from_id": 4, "to_id": 5, "initial_score": 1.0},
        ]

        propagated = module.propagate_connectivity_scores(links, iterations=2)

        self.assertGreater(propagated[2], 1.0)
        self.assertEqual(propagated[3], 1.0)


if __name__ == "__main__":
    unittest.main()
