import pytest
from logger.Logger import Logger
from flask_server.offloading.offloading_algo import OffloadingAlgo

# Initialize logger for testing
logger = Logger(config_path='./logger/logger_config.json').get_logger(__name__)


@pytest.fixture
def offloading_algo_instance():
    # Sample data for initialization
    avg_speed = 10.0    # Average network speed
    num_layers = 3  # Layers in the neural network
    layers_sizes = [10, 20, 15, 30]  # Sample layer sizes
    inference_time_device = [1.0, 2.0, 1.5, 3.0]  # Sample device inference times
    inference_time_edge = [0.5, 1.0, 0.8, 1.2]  # Sample edge inference times

    return OffloadingAlgo(avg_speed, num_layers, layers_sizes, inference_time_device, inference_time_edge)


def test_evaluation(offloading_algo_instance):
    # Test the evaluation method
    initial_cost = 2
    layer_data_size = 15
    edge_computation_cost = sum([0.5, 1.0, 0.8])

    result = offloading_algo_instance.evaluation(initial_cost, layer_data_size, edge_computation_cost, 10.0)

    assert isinstance(result, float)
    assert result > 0


def test_edge_only_computation_evaluation(offloading_algo_instance, caplog):
    # Test edge_only_computation_evaluation method
    offloading_algo_instance.edge_only_computation_evaluation()

    assert offloading_algo_instance.lowest_evaluation != float('inf')
    assert "Performing Edge Only Offloading:" in [rec.message for rec in caplog.records]


def test_mixed_computation_evaluation(offloading_algo_instance, caplog):
    # Test mixed_computation_evaluation method
    offloading_algo_instance.mixed_computation_evaluation()

    assert offloading_algo_instance.lowest_evaluation != float('inf')
    assert "Performing Partial Offloading:" in [rec.message for rec in caplog.records]


def test_device_only_evaluation(offloading_algo_instance, caplog):
    # Test device_only_evaluation method
    offloading_algo_instance.device_only_evaluation()

    assert offloading_algo_instance.lowest_evaluation != float('inf')
    assert "Performing Device Only Offloading:" in [rec.message for rec in caplog.records]


def test_static_offloading(offloading_algo_instance, caplog):
    # Test static_offloading method
    result = offloading_algo_instance.static_offloading()

    assert isinstance(result, int)
    assert 0 <= result <= offloading_algo_instance.num_layers
    assert "Computation Completed - Offloading Summary:" in [rec.message for rec in caplog.records]


if __name__ == "__main__":
    pytest.main()
