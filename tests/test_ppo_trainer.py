import csv
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from chargingpilot.environment.models import EpisodeData, SplitChargingEnvConfig, VehicleRequest
from chargingpilot.environment.split_charging_env import SplitChargingRequestEnv
from chargingpilot.cli import main as train_main
from chargingpilot.cli import parse_args, run_training
from chargingpilot.simulator.models import StationSpec, VehicleSpec
from chargingpilot.trainer.ppo_trainer import MaskedActorCritic, PPOTrainer, PPOTrainerConfig


class ZeroNetwork:
    def path_time(self, u: int, v: int, t: float, route_nodes=None) -> float:
        return 0.0

    def path_energy(self, u: int, v: int, t: float, vehicle_or_rho, route_nodes=None) -> float:
        return 0.0


def make_vehicle(vehicle_id: int) -> VehicleRequest:
    spec = VehicleSpec(
        battery_capacity=50.0,
        initial_soc=0.4,
        soc_min=0.0,
        p_max_kw=100.0,
        p_min_kw=20.0,
        rho_kwh_per_km=0.1,
        origin=1,
        destination=1,
        departure_time=0.0,
        path_nodes=(1,),
        path_edges=(),
        candidate_stations=(1,),
    )
    return VehicleRequest(
        vehicle_id=vehicle_id,
        decision_time=float(vehicle_id - 1),
        vehicle_spec=spec,
        target_soc=0.5,
    )


class PPOTrainerTests(unittest.TestCase):
    def test_ppo_trainer_module_does_not_configure_swanlab(self) -> None:
        source = Path("src/chargingpilot/trainer/ppo_trainer.py").read_text(encoding="utf-8").lower()

        self.assertNotIn("swanlab", source)

    def test_popart_update_preserves_raw_value_prediction(self) -> None:
        policy = MaskedActorCritic(
            obs_dim=3,
            action_dim=2,
            hidden_dim=4,
            use_popart=True,
            popart_beta=0.0,
            popart_epsilon=1e-5,
        )
        obs = torch.tensor([[0.2, -0.1, 0.5]], dtype=torch.float32)
        with torch.no_grad():
            _logits, before = policy.forward_raw_value(obs)

        policy.update_popart(torch.tensor([10.0, 14.0, 18.0], dtype=torch.float32))

        with torch.no_grad():
            _logits, after = policy.forward_raw_value(obs)

        self.assertGreater(float(policy.popart_std.item()), 0.0)
        self.assertNotEqual(float(policy.popart_mean.item()), 0.0)
        self.assertTrue(torch.allclose(before, after, atol=1e-5, rtol=1e-5))

    def test_trainer_runs_one_update_without_swanlab(self) -> None:
        episode = EpisodeData(
            station_specs=(
                StationSpec(station_id=1, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
            ),
            vehicle_requests=(make_vehicle(1), make_vehicle(2)),
            network=ZeroNetwork(),
            timestep_minutes=1.0,
        )
        env = SplitChargingRequestEnv(
            episode_factory=lambda: episode,
            config=SplitChargingEnvConfig(max_station_count=1),
        )
        trainer = PPOTrainer(
            env=env,
            config=PPOTrainerConfig(
                total_updates=1,
                episodes_per_update=1,
                batch_size=2,
                update_epochs=1,
                hidden_dim=16,
                learning_rate=1e-3,
            ),
        )
        before = [param.detach().clone() for param in trainer.policy.parameters()]

        metrics = trainer.train()

        self.assertEqual(metrics["updates"], 1)
        self.assertIn("policy_loss", metrics)
        changed = any(
            not torch.equal(old, new)
            for old, new in zip(before, trainer.policy.parameters())
        )
        self.assertTrue(changed)

    def test_trainer_runs_one_update_with_popart(self) -> None:
        episode = EpisodeData(
            station_specs=(
                StationSpec(station_id=1, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
            ),
            vehicle_requests=(make_vehicle(1), make_vehicle(2)),
            network=ZeroNetwork(),
            timestep_minutes=1.0,
        )
        env = SplitChargingRequestEnv(
            episode_factory=lambda: episode,
            config=SplitChargingEnvConfig(max_station_count=1),
        )
        trainer = PPOTrainer(
            env=env,
            config=PPOTrainerConfig(
                total_updates=1,
                episodes_per_update=1,
                batch_size=2,
                update_epochs=1,
                hidden_dim=16,
                learning_rate=1e-3,
                use_popart=True,
            ),
        )

        metrics = trainer.train()

        self.assertEqual(metrics["updates"], 1)
        self.assertIn("popart_mean", metrics)
        self.assertIn("popart_std", metrics)
        self.assertGreater(metrics["popart_std"], 0.0)

    def test_gae_can_follow_master_transition_target_instead_of_chronological_neighbor(self) -> None:
        episode = EpisodeData(
            station_specs=(
                StationSpec(station_id=1, charge_capacity=1, p_plug_kw=100.0, p_max_kw=100.0, eta=1.0),
            ),
            vehicle_requests=(make_vehicle(1),),
            network=ZeroNetwork(),
            timestep_minutes=1.0,
        )
        env = SplitChargingRequestEnv(
            episode_factory=lambda: episode,
            config=SplitChargingEnvConfig(max_station_count=1),
        )
        trainer = PPOTrainer(
            env=env,
            config=PPOTrainerConfig(
                gamma=0.9,
                gae_lambda=0.5,
                hidden_dim=16,
            ),
        )

        advantages, returns = trainer._compute_gae(
            rewards=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            values=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            next_values=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            dones=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            next_indices=[2, None, None],
        )

        torch.testing.assert_close(advantages, torch.tensor([0.45, 0.0, 1.0]))
        torch.testing.assert_close(returns, torch.tensor([0.45, 0.0, 1.0]))

    def test_train_ppo_cli_defaults_to_full_data_workflow(self) -> None:
        args = parse_args([])

        self.assertTrue(args.use_data_factory)
        self.assertTrue(args.use_popart)
        self.assertTrue(args.use_swanlab)
        self.assertEqual(args.data_dir, Path("datasets/train"))
        self.assertEqual(args.station_setting, Path("exps/data/setting_7stations_pv_ess.yaml"))
        self.assertEqual(args.travel_time_model, Path("models/highway_travel_time_gct.pt"))
        self.assertEqual(args.output, Path("models/ppo_split_charging.pt"))
        self.assertEqual(args.checkpoint_interval, 5000)

    def test_train_ppo_cli_can_disable_default_integrations(self) -> None:
        args = parse_args(["--demo", "--no-popart", "--no-swanlab", "--no-travel-time-model"])

        self.assertFalse(args.use_data_factory)
        self.assertFalse(args.use_popart)
        self.assertFalse(args.use_swanlab)
        self.assertIsNone(args.travel_time_model)

    def test_train_ppo_cli_accepts_online_and_api_key(self) -> None:
        args = parse_args(["--online", "--api-key", "test-key"])

        self.assertTrue(args.online)
        self.assertEqual(args.api_key, "test-key")

    def test_train_ppo_popart_cli_legacy_enable_flag_is_accepted(self) -> None:
        args = parse_args(["--use-popart"])

        self.assertTrue(args.use_popart)

    def test_train_ppo_data_factory_cli_accepts_travel_time_model(self) -> None:
        args = parse_args(
            [
                "--use-data-factory",
                "--data-dir",
                "custom-data",
                "--station-setting",
                "custom-setting.yaml",
                "--data-seed",
                "19",
                "--travel-time-model",
                "models/custom.pt",
            ]
        )

        self.assertTrue(args.use_data_factory)
        self.assertEqual(str(args.data_dir), "custom-data")
        self.assertEqual(str(args.station_setting), "custom-setting.yaml")
        self.assertEqual(args.data_seed, 19)
        self.assertEqual(str(args.travel_time_model), "models\\custom.pt")

    def test_run_training_saves_checkpoint_and_logs_to_swanlab(self) -> None:
        fake_swanlab = FakeSwanLab()
        original_swanlab = sys.modules.get("swanlab")
        sys.modules["swanlab"] = fake_swanlab
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data_dir, setting_path = write_minimal_training_dataset(root)
                output_path = root / "ppo.pt"
                args = parse_args(
                    [
                        "--data-dir",
                        str(data_dir),
                        "--station-setting",
                        str(setting_path),
                        "--no-travel-time-model",
                        "--output",
                        str(output_path),
                        "--total-updates",
                        "1",
                        "--episodes-per-update",
                        "1",
                        "--batch-size",
                        "2",
                        "--update-epochs",
                        "1",
                        "--swanlab-mode",
                        "offline",
                    ]
                )

                result = run_training(args)

                checkpoint = torch.load(output_path, map_location="cpu")
        finally:
            if original_swanlab is None:
                sys.modules.pop("swanlab", None)
            else:
                sys.modules["swanlab"] = original_swanlab

        self.assertEqual(result.checkpoint_path, output_path)
        self.assertEqual(result.metrics["updates"], 1)
        self.assertIn("popart_mean", result.metrics)
        self.assertIn("popart_std", result.metrics)
        self.assertIn("policy_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("trainer_config", checkpoint)
        self.assertIn("metrics", checkpoint)
        self.assertIn("popart_mean", checkpoint["metrics"])
        self.assertEqual(checkpoint["trainer_config"]["use_popart"], True)
        self.assertEqual(fake_swanlab.init_calls[0]["mode"], "offline")
        self.assertEqual(len(fake_swanlab.logs), 1)
        self.assertEqual(fake_swanlab.logs[0][1], 1)
        self.assertEqual(fake_swanlab.finish_count, 1)

    def test_run_training_logs_into_swanlab_cloud_with_api_key(self) -> None:
        fake_swanlab = FakeSwanLab()
        original_swanlab = sys.modules.get("swanlab")
        sys.modules["swanlab"] = fake_swanlab
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data_dir, setting_path = write_minimal_training_dataset(root)
                args = parse_args(
                    [
                        "--data-dir",
                        str(data_dir),
                        "--station-setting",
                        str(setting_path),
                        "--no-travel-time-model",
                        "--output",
                        str(root / "ppo.pt"),
                        "--total-updates",
                        "1",
                        "--episodes-per-update",
                        "1",
                        "--batch-size",
                        "2",
                        "--update-epochs",
                        "1",
                        "--online",
                        "--api-key",
                        "test-key",
                    ]
                )

                run_training(args)
        finally:
            if original_swanlab is None:
                sys.modules.pop("swanlab", None)
            else:
                sys.modules["swanlab"] = original_swanlab

        self.assertEqual(fake_swanlab.login_calls, [{"api_key": "test-key", "save": False}])
        self.assertEqual(fake_swanlab.init_calls[0]["mode"], "cloud")

    def test_run_training_prints_update_metrics_and_writes_periodic_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, setting_path = write_minimal_training_dataset(root)
            output_path = root / "ppo.pt"
            args = parse_args(
                [
                    "--data-dir",
                    str(data_dir),
                    "--station-setting",
                    str(setting_path),
                    "--no-travel-time-model",
                    "--no-swanlab",
                    "--output",
                    str(output_path),
                    "--total-updates",
                    "2",
                    "--episodes-per-update",
                    "1",
                    "--batch-size",
                    "2",
                    "--update-epochs",
                    "1",
                    "--checkpoint-interval",
                    "1",
                ]
            )
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                run_training(args)

            output = stream.getvalue()
            self.assertIn("training update 1:", output)
            self.assertIn("training update 2:", output)
            self.assertIn("policy_loss=", output)
            self.assertTrue((root / "ppo_update_000001.pt").is_file())
            self.assertTrue((root / "ppo_update_000002.pt").is_file())

    def test_main_prints_training_status_prompts(self) -> None:
        original_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sys.argv = [
                    "train_ppo.py",
                    "--demo",
                    "--no-swanlab",
                    "--no-popart",
                    "--output",
                    str(root / "ppo.pt"),
                    "--total-updates",
                    "1",
                    "--episodes-per-update",
                    "1",
                    "--batch-size",
                    "8",
                    "--update-epochs",
                    "1",
                    "--checkpoint-interval",
                    "0",
                ]
                stream = io.StringIO()

                with contextlib.redirect_stdout(stream):
                    train_main()

                output = stream.getvalue()
        finally:
            sys.argv = original_argv

        self.assertIn("Start Training", output)
        self.assertIn("Training Finished", output)
        self.assertIn("checkpoint:", output)


class FakeSwanLab(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("swanlab")
        self.init_calls: list[dict] = []
        self.login_calls: list[dict] = []
        self.logs: list[tuple[dict[str, float], int | None]] = []
        self.finish_count = 0

    def login(self, **kwargs) -> None:
        self.login_calls.append(kwargs)

    def init(self, **kwargs) -> None:
        self.init_calls.append(kwargs)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        self.finish_count += 1


def write_minimal_training_dataset(root: Path) -> tuple[Path, Path]:
    data_dir = root / "episodes"
    data_dir.mkdir()
    setting_path = root / "setting.yaml"
    setting_path.write_text(
        "\n".join(
            [
                "scenario_name: ppo_test",
                "station_ids: [1]",
                "stations:",
                "  charge_capacity: [1]",
                "  p_plug_kw: 120.0",
                "  p_max_kw: [120.0]",
                "  p_grid_max_kw: [120.0]",
                "  eta: 1.0",
                "environment:",
                "  timestep_minutes: 1.0",
                "  episode_horizon_minutes: 1440.0",
                "  max_station_count: 1",
                "  max_power_kw: 500.0",
                "  max_ess_kwh: 10000.0",
            ]
        ),
        encoding="utf-8",
    )
    episode_path = data_dir / "episode_0001.csv"
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
        for vehicle_id, arrival_time in ((1, "00:00"), (2, "00:01")):
            writer.writerow(
                {
                    "episode_id": 1,
                    "vehicle_id": f"EV{vehicle_id:08d}",
                    "arrival_time": arrival_time,
                    "start_soc": "0.30",
                    "target_soc": "0.60",
                    "B_i": "50",
                    "o_i": "S1",
                    "d_i": "S1",
                    "rho_i": "0.2",
                }
            )
    return data_dir, setting_path


if __name__ == "__main__":
    unittest.main()
