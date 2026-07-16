import csv
import math
import tempfile
import unittest
from pathlib import Path

import torch

from chargingpilot.environment.data_factory import DataFactory, DataFactoryConfig
from chargingpilot.network.GCT import GraphConvolutionalTransformer, TravelTimeModelConfig


class DataFactoryTests(unittest.TestCase):
    def test_station_specs_apply_pv_indicator_and_keep_ess_indicator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_indicator_dataset(root)

            factory = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    shuffle=False,
                )
            )
            episode = factory()

        specs = episode.station_specs
        self.assertEqual([spec.station_id for spec in specs], [1, 2])
        self.assertEqual(specs[0].renewable_power_trace, ((0.0, 0.0), (1.0, 100.0)))
        self.assertIsNone(specs[1].renewable_power_trace)
        self.assertEqual([spec.ess_capacity_kwh for spec in specs], [0.0, 6000.0])
        self.assertEqual([spec.p_ess_discharge_max_kw for spec in specs], [0.0, 3000.0])

    def test_station_specs_require_pv_indicator_only_when_pv_csv_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_indicator_dataset(root)
            settings = setting_path.read_text(encoding="utf-8")
            without_indicator = settings.replace("  pv_indicator: [1, 0]\n", "")
            setting_path.write_text(without_indicator, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "pv_indicator.*required"):
                DataFactory(
                    DataFactoryConfig(
                        episodes_dir=episodes_dir,
                        station_setting_path=setting_path,
                        shuffle=False,
                    )
                )

            setting_path.write_text(
                "\n".join(
                    line
                    for line in without_indicator.splitlines()
                    if not line.strip().startswith("pv_power_csv:")
                ),
                encoding="utf-8",
            )
            factory = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    shuffle=False,
                )
            )
            episode = factory()

        self.assertTrue(all(spec.renewable_power_trace is None for spec in episode.station_specs))

    def test_episode_csv_rows_become_bidirectional_vehicle_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_minimal_dataset(root)

            episode = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    shuffle=False,
                )
            )()

        first, second = episode.vehicle_requests
        self.assertEqual(first.vehicle_id, 1)
        self.assertEqual(first.decision_time, 30.0)
        self.assertEqual(first.vehicle_spec.origin, 1)
        self.assertEqual(first.vehicle_spec.destination, 3)
        self.assertEqual(first.vehicle_spec.path_nodes, (1, 2, 3))
        self.assertEqual(first.vehicle_spec.candidate_stations, (1, 2))
        self.assertAlmostEqual(first.vehicle_spec.initial_soc, 0.2)
        self.assertAlmostEqual(first.vehicle_spec.soc_min, 0.0)
        self.assertAlmostEqual(first.target_soc, 0.9)
        self.assertFalse(hasattr(first.vehicle_spec, "soc_max"))
        self.assertAlmostEqual(first.vehicle_spec.battery_capacity, 50.0)
        self.assertAlmostEqual(first.vehicle_spec.rho_kwh_per_km, 0.2)

        self.assertEqual(second.vehicle_spec.origin, 3)
        self.assertEqual(second.vehicle_spec.destination, 1)
        self.assertEqual(second.vehicle_spec.path_nodes, (3, 2, 1))
        self.assertEqual(second.vehicle_spec.candidate_stations, (3, 2))

        forward_time = episode.network.path_time(1, 3, 30.0, route_nodes=(1, 2, 3))
        reverse_time = episode.network.path_time(3, 1, 30.0, route_nodes=(3, 2, 1))
        self.assertGreater(forward_time, 0.0)
        self.assertAlmostEqual(forward_time, reverse_time)
        self.assertAlmostEqual(
            episode.network.path_energy(1, 3, 30.0, first.vehicle_spec, route_nodes=(1, 2, 3)),
            10.0,
        )

    def test_seeded_shuffle_cycle_is_reproducible_and_covers_all_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_minimal_dataset(root, episode_count=3)

            first_factory = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    seed=17,
                    shuffle=True,
                )
            )
            second_factory = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    seed=17,
                    shuffle=True,
                )
            )
            first_sequence = [first_factory().vehicle_requests[0].vehicle_id for _ in range(4)]
            second_sequence = [second_factory().vehicle_requests[0].vehicle_id for _ in range(4)]

        self.assertEqual(first_sequence, second_sequence)
        self.assertEqual(set(first_sequence[:3]), {1, 3, 5})

    def test_manifest_selects_split_preserves_eval_order_and_only_shuffles_train_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_indicator_dataset(root)

            def write_day(filename: str, base_vehicle_id: int) -> None:
                with (episodes_dir / filename).open("w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=[
                            "vehicle_id",
                            "arrival_time",
                            "start_soc",
                            "target_soc",
                            "B_i",
                            "o_i",
                            "d_i",
                            "rho_i",
                        ],
                    )
                    writer.writeheader()
                    for vehicle_id, arrival_time in (
                        (base_vehicle_id + 2, "00:30"),
                        (base_vehicle_id + 1, "00:10"),
                        (base_vehicle_id, "00:10"),
                    ):
                        writer.writerow(
                            {
                                "vehicle_id": f"EV{vehicle_id:08d}",
                                "arrival_time": arrival_time,
                                "start_soc": "0.2",
                                "target_soc": "0.9",
                                "B_i": "50",
                                "o_i": "S1",
                                "d_i": "S2",
                                "rho_i": "0.2",
                            }
                        )

            day_specs = (
                ("train_a.csv", 100),
                ("train_b.csv", 200),
                ("train_c.csv", 300),
                ("validation_a.csv", 400),
                ("validation_b.csv", 500),
                ("test_a.csv", 600),
            )
            for filename, base_vehicle_id in day_specs:
                write_day(filename, base_vehicle_id)
            manifest_path = root / "request_split.yaml"
            manifest_path.write_text(
                "train: [train_a.csv, train_b.csv, train_c.csv]\n"
                "validation: [validation_b.csv, validation_a.csv]\n"
                "test: [test_a.csv]\n",
                encoding="utf-8",
            )

            def make_factory(split: str) -> DataFactory:
                return DataFactory(
                    DataFactoryConfig(
                        episodes_dir=episodes_dir,
                        station_setting_path=setting_path,
                        request_manifest_path=manifest_path,
                        request_split=split,
                        seed=7,
                        shuffle=True,
                    )
                )

            validation = make_factory("validation")
            validation_days = [validation(), validation()]
            test_day = make_factory("test")()
            first_train = make_factory("train")
            second_train = make_factory("train")
            first_train_days = [first_train(), first_train(), first_train()]
            second_train_days = [second_train(), second_train(), second_train()]

        self.assertEqual(
            [[request.vehicle_id for request in day.vehicle_requests] for day in validation_days],
            [[500, 501, 502], [400, 401, 402]],
        )
        self.assertEqual([request.vehicle_id for request in test_day.vehicle_requests], [600, 601, 602])
        first_train_order = [day.vehicle_requests[0].vehicle_id for day in first_train_days]
        second_train_order = [day.vehicle_requests[0].vehicle_id for day in second_train_days]
        self.assertEqual(first_train_order, [300, 100, 200])
        self.assertEqual(second_train_order, first_train_order)
        self.assertEqual(
            [[request.vehicle_id for request in day.vehicle_requests] for day in first_train_days],
            [[300, 301, 302], [100, 101, 102], [200, 201, 202]],
        )

    def test_travel_time_model_checkpoint_is_used_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_minimal_dataset(root)
            model_path = root / "travel_time.pt"
            self._write_constant_travel_time_checkpoint(model_path, value=12.5)

            episode = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    travel_time_model_path=model_path,
                    shuffle=False,
                )
            )()

        self.assertAlmostEqual(
            episode.network.path_time(1, 3, 30.0, route_nodes=(1, 2, 3)),
            12.5,
        )

    def test_road_map_vdf_network_uses_request_path_nodes_for_time_and_energy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes_dir, setting_path = self._write_minimal_dataset(
                root,
                station_ids=(1, 2, 4),
                row_overrides=[
                    {
                        "vehicle_id": "EV00000001",
                        "arrival_time": "08:00",
                        "start_soc": "0.5",
                        "target_soc": "0.9",
                        "B_i": "50",
                        "o_i": "1",
                        "d_i": "4",
                        "rho_i": "0.2",
                        "path_node_ids": "1-2-4",
                    }
                ],
            )
            nodes_path, links_path, flows_path, parameters_path = self._write_vdf_road_map(root)

            episode = DataFactory(
                DataFactoryConfig(
                    episodes_dir=episodes_dir,
                    station_setting_path=setting_path,
                    use_vdf_road_map=True,
                    road_map_nodes_path=nodes_path,
                    road_map_links_path=links_path,
                    link_flows_path=flows_path,
                    link_parameters_path=parameters_path,
                    shuffle=False,
                )
            )()

            request = episode.vehicle_requests[0]
            self.assertEqual(request.vehicle_spec.path_nodes, (1, 2, 4))
            self.assertEqual(request.vehicle_spec.candidate_stations, (1, 2))
            travel_time = episode.network.path_time(1, 4, 8 * 60.0, route_nodes=(1, 2, 4))
            self.assertTrue(math.isclose(travel_time, 2.0))
            self.assertTrue(
                math.isclose(
                    episode.network.path_energy(1, 4, 8 * 60.0, request.vehicle_spec, route_nodes=(1, 2, 4)),
                    0.4,
                )
            )

    def _write_indicator_dataset(self, root: Path) -> tuple[Path, Path]:
        episodes_dir = root / "episodes"
        episodes_dir.mkdir()
        pv_path = root / "pv.csv"
        pv_path.write_text("time,pv_power_kw\n00:00,0.0\n00:01,100.0\n", encoding="utf-8")
        setting_path = root / "setting.yaml"
        setting_path.write_text(
            "\n".join(
                [
                    "station_ids: [1, 2]",
                    "stations:",
                    "  charge_capacity: [1, 2]",
                    "  p_plug_kw: 180.0",
                    "  p_max_kw: [180.0, 360.0]",
                    "  p_grid_max_kw: [180.0, 360.0]",
                    "  eta: 0.95",
                    "renewable:",
                    "  pv_indicator: [1, 0]",
                    f"  pv_power_csv: {pv_path.name}",
                    "ess:",
                    "  ess_indicator: [0, 1]",
                    "  capacity_kwh: 6000.0",
                    "  initial_kwh: 0.0",
                    "  p_charge_max_kw: 3000.0",
                    "  p_discharge_max_kw: 3000.0",
                ]
            ),
            encoding="utf-8",
        )
        episode_path = episodes_dir / "episode_0001.csv"
        with episode_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "episode_id",
                    "vehicle_id",
                    "arrival_time",
                    "start_soc",
                    "target_soc",
                    "B_i",
                    "o_i",
                    "d_i",
                    "rho_i",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "episode_id": 1,
                    "vehicle_id": "EV00000001",
                    "arrival_time": "00:30",
                    "start_soc": "0.2",
                    "target_soc": "0.9",
                    "B_i": "50",
                    "o_i": "S1",
                    "d_i": "S2",
                    "rho_i": "0.2",
                }
            )
        return episodes_dir, setting_path

    def _write_minimal_dataset(
        self,
        root: Path,
        *,
        episode_count: int = 1,
        station_ids: tuple[int, ...] = (1, 2, 3),
        row_overrides: list[dict[str, str]] | None = None,
    ) -> tuple[Path, Path]:
        episodes_dir = root / "episodes"
        episodes_dir.mkdir()
        pv_path = root / "pv.csv"
        pv_path.write_text("time,pv_power_kw\n00:00,0.0\n00:01,100.0\n", encoding="utf-8")
        setting_path = root / "setting.yaml"
        setting_path.write_text(
            "\n".join(
                [
                    "scenario_name: test",
                    f"station_ids: [{', '.join(str(item) for item in station_ids)}]",
                    "stations:",
                    f"  charge_capacity: [{', '.join(str(index + 1) for index in range(len(station_ids)))}]",
                    "  p_plug_kw: 180.0",
                    f"  p_max_kw: [{', '.join(str(180.0 * (index + 1)) for index in range(len(station_ids)))}]",
                    f"  p_grid_max_kw: [{', '.join(str(180.0 * (index + 1)) for index in range(len(station_ids)))}]",
                    "  eta: 0.95",
                    "renewable:",
                    f"  pv_indicator: [{', '.join('0' for _ in station_ids)}]",
                    f"  pv_power_csv: {pv_path.name}",
                    "  time_column: time",
                    "  pv_power_column: pv_power_kw",
                    "ess:",
                    f"  ess_indicator: [{', '.join('1' if index == 1 else '0' for index, _ in enumerate(station_ids))}]",
                    "  capacity_kwh: 6000.0",
                    "  initial_kwh: 0.0",
                    "  p_charge_max_kw: 3000.0",
                    "  p_discharge_max_kw: 3000.0",
                    "  charge_efficiency: 0.96",
                    "  discharge_efficiency: 0.96",
                    "  power_trace: null",
                    "environment:",
                    "  timestep_minutes: 1.0",
                    "  episode_horizon_minutes: 1440.0",
                    "  max_station_count: 3",
                    "  max_power_kw: 10000.0",
                    "  max_ess_kwh: 6000.0",
                ]
            ),
            encoding="utf-8",
        )
        rows_override = row_overrides
        for episode_id in range(1, episode_count + 1):
            path = episodes_dir / f"episode_{episode_id:04d}.csv"
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "episode_id",
                        "vehicle_id",
                        "arrival_time",
                        "start_soc",
                        "target_soc",
                        "B_i",
                        "o_i",
                        "d_i",
                        "rho_i",
                        "path_node_ids",
                    ],
                )
                writer.writeheader()
                base_vehicle = (episode_id - 1) * 2 + 1
                if rows_override is not None:
                    for row in rows_override:
                        writer.writerow({"episode_id": episode_id, **row})
                else:
                    writer.writerow(
                        {
                            "episode_id": episode_id,
                            "vehicle_id": f"EV{base_vehicle:08d}",
                            "arrival_time": "00:30",
                            "start_soc": "0.2",
                            "target_soc": "0.9",
                            "B_i": "50",
                            "o_i": f"S{station_ids[0]}",
                            "d_i": f"S{station_ids[-1]}",
                            "rho_i": "0.2",
                            "path_node_ids": "",
                        }
                    )
                    writer.writerow(
                        {
                            "episode_id": episode_id,
                            "vehicle_id": f"EV{base_vehicle + 1:08d}",
                            "arrival_time": "00:45",
                            "start_soc": "0.3",
                            "target_soc": "0.8",
                            "B_i": "60",
                            "o_i": f"S{station_ids[-1]}",
                            "d_i": f"S{station_ids[0]}",
                            "rho_i": "0.18",
                            "path_node_ids": "",
                        }
                    )
        return episodes_dir, setting_path

    def _write_vdf_road_map(self, root: Path) -> tuple[Path, Path, Path, Path]:
        nodes_path = root / "nodes.geojson"
        links_path = root / "links.geojson"
        flows_path = root / "link_24h_flows.csv"
        parameters_path = root / "link_parameters.csv"
        nodes_path.write_text(
            "\n".join(
                [
                    '{"type":"FeatureCollection","features":[',
                    '{"type":"Feature","properties":{"id":1,"type":"toll"},"geometry":{"type":"Point","coordinates":[1,0]}},',
                    '{"type":"Feature","properties":{"id":2,"type":"service"},"geometry":{"type":"Point","coordinates":[2,0]}},',
                    '{"type":"Feature","properties":{"id":4,"type":"toll"},"geometry":{"type":"Point","coordinates":[4,0]}}',
                    "]}",
                ]
            ),
            encoding="utf-8",
        )
        links_path.write_text(
            "\n".join(
                [
                    '{"type":"FeatureCollection","features":[',
                    '{"type":"Feature","properties":{"from_id":1,"to_id":2,"length":1000},"geometry":{"type":"LineString","coordinates":[[1,0],[2,0]]}},',
                    '{"type":"Feature","properties":{"from_id":2,"to_id":4,"length":1000},"geometry":{"type":"LineString","coordinates":[[2,0],[4,0]]}}',
                    "]}",
                ]
            ),
            encoding="utf-8",
        )
        flows_path.write_text(
            "link_id,from_id,to_id,flow_h08\n"
            "1,1,2,0\n"
            "2,2,4,0\n",
            encoding="utf-8",
        )
        parameters_path.write_text(
            "link_id,from_id,to_id,length_m,lane_count,free_flow_speed_kmph,capacity_veh_per_hour,m\n"
            "1,1,2,1000,1,60,1000,2.5\n"
            "2,2,4,1000,1,60,1000,2.5\n",
            encoding="utf-8",
        )
        return nodes_path, links_path, flows_path, parameters_path

    def _write_constant_travel_time_checkpoint(self, path: Path, *, value: float) -> None:
        config = TravelTimeModelConfig(
            segment_feature_dim=5,
            max_route_len=6,
            departure_feature_dim=5,
            embedding_dim=8,
            num_heads=2,
            transformer_layers=1,
            feedforward_dim=16,
            hidden_dim=4,
            dropout=0.0,
        )
        model = GraphConvolutionalTransformer(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output_layers[-1].bias.fill_(float(value))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": config.__dict__,
                "station_chain": [1, 2, 3, 4, 5, 6, 7],
                "segment_ids": ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7"],
            },
            path,
        )


if __name__ == "__main__":
    unittest.main()
