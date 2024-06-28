import pytest

from flask_server.configs.configs import OffloadingManagerConfigs
from flask_server.offloading.offloading_manager import OffloadingManager


def test_offloading_without_model_name():
    assert ValueError("Error: parameter [model_name] cannot be None")


def test_offloading_without_algorithm_version():
    offloading_manager = OffloadingManager(algorithm_version=None)
    assert offloading_manager.algorithm_version in OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION


def test_offloading_with_invalid_working_strategy():
    with pytest.raises(ValueError) as exec_info:
        OffloadingManager(working_strategy="invalid_strategy")
    assert str(exec_info.value) == (f"Error: The given [working_strategy] is not valid. "
                                    f"Allowed strategies are: {OffloadingManagerConfigs.ALLOWED_WORKING_STRATEGIES}")


def test_offloading_with_invalid_start_layer_index():
    with pytest.raises(ValueError) as exec_info:
        OffloadingManager(start_layer_index=-1)
    assert str(exec_info.value) == "Error: parameter [start_layer_index] must be a non-negative integer"


def test_offloading_with_model_name():
    model_name = "basic_model"
    offloading_tool = OffloadingManager(model_name=model_name)
    algorithm_version = OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION
    assert offloading_tool.offload() == f"Offloading [{algorithm_version}]: {model_name}"


def test_offloading_with_custom_algorithm_version():
    model_name = "basic_model"
    algorithm_version = "v2.0"
    offloading_tool = OffloadingManager(model_name=model_name, algorithm_version=algorithm_version)
    assert offloading_tool.offload() == f"Offloading [{algorithm_version}]: {model_name}"


def test_offloading_with_custom_working_strategy():
    model_name = "basic_model"
    working_strategy = OffloadingManagerConfigs.ALLOWED_WORKING_STRATEGIES[0]
    offloading_tool = OffloadingManager(model_name=model_name, working_strategy=working_strategy)
    algorithm_version = OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION
    assert offloading_tool.working_strategy == working_strategy
    assert offloading_tool.offload() == f"Offloading [{algorithm_version}]: {model_name}"


def test_offloading_with_custom_start_layer_index():
    model_name = "basic_model"
    start_layer_index = 5
    offloading_tool = OffloadingManager(model_name=model_name, start_layer_index=start_layer_index)
    algorithm_version = OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION
    assert offloading_tool.start_layer_index == start_layer_index
    assert offloading_tool.offload() == f"Offloading [{algorithm_version}]: {model_name}"


if __name__ == "__main__":
    pytest.main()
