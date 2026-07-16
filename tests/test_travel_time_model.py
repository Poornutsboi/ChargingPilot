import importlib
import unittest

import torch

from chargingpilot.network import (
    GraphConvolutionLayer,
    GraphConvolutionalTransformer,
    TravelTimeBatch,
    TravelTimeModelConfig,
    predict_travel_time,
)


class TravelTimeModelTests(unittest.TestCase):
    def test_gct_module_name_is_importable(self) -> None:
        module = importlib.import_module("chargingpilot.network.GCT")

        self.assertIs(module.GraphConvolutionalTransformer, GraphConvolutionalTransformer)

    def test_forward_returns_one_prediction_per_route(self) -> None:
        torch.manual_seed(7)
        model = GraphConvolutionalTransformer(
            TravelTimeModelConfig(
                segment_feature_dim=6,
                max_route_len=4,
                embedding_dim=8,
                num_heads=2,
                transformer_layers=3,
                feedforward_dim=16,
                hidden_dim=12,
                dropout=0.0,
            )
        )
        batch = TravelTimeBatch(
            segment_features=torch.randn(5, 6),
            edge_index=torch.tensor(
                [
                    [0, 1, 2, 3],
                    [1, 2, 3, 4],
                ],
                dtype=torch.long,
            ),
            route_segment_ids=torch.tensor(
                [
                    [0, 1, 2, 0],
                    [1, 3, 4, 0],
                ],
                dtype=torch.long,
            ),
            route_mask=torch.tensor(
                [
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ),
        )

        predictions = model(batch)

        self.assertEqual(tuple(predictions.shape), (2,))

    def test_graph_convolution_aggregates_upstream_and_downstream_neighbors(self) -> None:
        layer = GraphConvolutionLayer(1, 1, bias=False)
        with torch.no_grad():
            layer.linear.weight.fill_(1.0)

        features = torch.tensor([[1.0], [10.0], [100.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        with torch.no_grad():
            output = layer(features, edge_index)

        self.assertAlmostEqual(float(output[0, 0]), 5.5)
        self.assertAlmostEqual(float(output[1, 0]), 37.0)
        self.assertAlmostEqual(float(output[2, 0]), 55.0)

    def test_padding_segments_do_not_change_predictions(self) -> None:
        torch.manual_seed(11)
        model = GraphConvolutionalTransformer(
            TravelTimeModelConfig(
                segment_feature_dim=4,
                max_route_len=4,
                embedding_dim=8,
                num_heads=2,
                transformer_layers=1,
                feedforward_dim=16,
                hidden_dim=10,
                dropout=0.0,
            )
        )
        model.eval()
        segment_features = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        route_mask = torch.tensor([[True, True, False, False]])

        base_batch = TravelTimeBatch(
            segment_features=segment_features,
            edge_index=edge_index,
            route_segment_ids=torch.tensor([[0, 1, 0, 0]], dtype=torch.long),
            route_mask=route_mask,
        )
        changed_padding_batch = TravelTimeBatch(
            segment_features=segment_features,
            edge_index=edge_index,
            route_segment_ids=torch.tensor([[0, 1, 3, 4]], dtype=torch.long),
            route_mask=route_mask,
        )

        with torch.no_grad():
            base_prediction = model(base_batch)
            changed_padding_prediction = model(changed_padding_batch)

        torch.testing.assert_close(base_prediction, changed_padding_prediction)

    def test_departure_features_are_supported(self) -> None:
        torch.manual_seed(13)
        model = GraphConvolutionalTransformer(
            TravelTimeModelConfig(
                segment_feature_dim=5,
                max_route_len=3,
                departure_feature_dim=2,
                embedding_dim=8,
                num_heads=2,
                transformer_layers=1,
                feedforward_dim=16,
                hidden_dim=10,
                dropout=0.0,
            )
        )
        batch = TravelTimeBatch(
            segment_features=torch.randn(4, 5),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            route_segment_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            route_mask=torch.tensor([[True, True, True]]),
            departure_features=torch.tensor([[8.0, 1.0]]),
        )

        predictions = model(batch)

        self.assertEqual(tuple(predictions.shape), (1,))

    def test_predict_travel_time_clamps_negative_inference_outputs(self) -> None:
        class NegativeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.training_state_seen = None

            def forward(self, batch: TravelTimeBatch) -> torch.Tensor:
                self.training_state_seen = bool(self.training)
                return torch.tensor([-3.0, 4.5])

        model = NegativeModel()
        model.train()
        batch = TravelTimeBatch(
            segment_features=torch.randn(2, 3),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            route_segment_ids=torch.tensor([[0], [1]], dtype=torch.long),
            route_mask=torch.tensor([[True], [True]]),
        )

        predictions = predict_travel_time(model, batch)

        torch.testing.assert_close(predictions, torch.tensor([0.0, 4.5]))
        self.assertFalse(model.training_state_seen)
        self.assertTrue(model.training)

    def test_predict_travel_time_minutes_is_not_exported(self) -> None:
        import chargingpilot.network as network

        self.assertFalse(hasattr(network, "predict_travel_time_minutes"))

    def test_config_rejects_attention_dimensions_that_do_not_divide_evenly(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedding_dim must be divisible"):
            TravelTimeModelConfig(
                segment_feature_dim=4,
                max_route_len=3,
                embedding_dim=10,
                num_heads=3,
            )


if __name__ == "__main__":
    unittest.main()
