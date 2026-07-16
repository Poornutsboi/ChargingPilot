import importlib.util
import json
import math
import tempfile
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "chargingpilot" / "roadmap" / "visualize_road_map.py"


def load_module():
    spec = importlib.util.spec_from_file_location("visualize_road_map", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoadMapVisualizationTests(unittest.TestCase):
    def test_web_mercator_to_lonlat_matches_known_origin(self):
        module = load_module()

        lon, lat = module.web_mercator_to_lonlat(0, 0)

        self.assertEqual(lon, 0)
        self.assertEqual(lat, 0)

    def test_web_mercator_to_lonlat_matches_zhejiang_sample(self):
        module = load_module()

        lon, lat = module.web_mercator_to_lonlat(13292738.099818543, 3505766.5124114174)

        self.assertTrue(math.isclose(lon, 119.4106979, abs_tol=1e-6))
        self.assertTrue(math.isclose(lat, 30.0172434, abs_tol=1e-6))

    def test_load_network_summary_counts_and_computes_null_lengths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            nodes_path = tmp_root / "nodes.geojson"
            links_path = tmp_root / "links.geojson"
            nodes_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"id": 1, "type": "toll"},
                                "geometry": {"type": "Point", "coordinates": [0, 0]},
                            },
                            {
                                "type": "Feature",
                                "properties": {"id": 2, "type": "service"},
                                "geometry": {"type": "Point", "coordinates": [100, 100]},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            links_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"from_id": 1, "to_id": 2, "length": None},
                                "geometry": {"type": "LineString", "coordinates": [[0, 0], [100, 100]]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            network = module.load_network(nodes_path, links_path)

        self.assertEqual(network["stats"]["node_count"], 2)
        self.assertEqual(network["stats"]["link_count"], 1)
        self.assertEqual(network["stats"]["node_type_counts"], {"toll": 1, "service": 1})
        self.assertEqual(network["links"][0]["length_source"], "computed_from_geometry")
        self.assertTrue(math.isclose(network["links"][0]["length"], math.sqrt(20000), abs_tol=1e-9))
        self.assertEqual(network["links"][0]["length_text"], "141 m")
        self.assertEqual(network["stats"]["missing_length_count"], 0)
        self.assertEqual(network["stats"]["computed_length_count"], 1)

    def test_html_includes_a_renewable_station_highlight_toggle(self):
        module = load_module()
        network = {
            "nodes": [],
            "links": [],
            "stats": {
                "node_count": 0,
                "link_count": 0,
                "node_type_counts": {"service": 2},
                "renewable_station_count": 1,
                "mercator_bbox": [0, 0, 1, 1],
                "lonlat_bbox": [0, 0, 1, 1],
                "total_length_km": 0.0,
            },
        }
        render_data = {"width": 1, "height": 1, "view_box": "0 0 1 1", "nodes": [], "links": [], "stats": network["stats"]}

        html = module._html_document(network, render_data, "<svg></svg>")

        self.assertIn('id="renewable-toggle"', html)
        self.assertIn("highlight-renewable", html)


if __name__ == "__main__":
    unittest.main()
