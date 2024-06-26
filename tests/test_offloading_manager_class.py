import pytest

from flask_server.configs.configs import OffloadingManagerConfigs
from flask_server.offloading.OffloadingManager import OffloadingManager


def test_offloading_without_model_name():
    assert ValueError("Error: parameter [model_name] cannot be None")


def test_offloading_with_model_name():
    offloading_tool = OffloadingManager()
    model_name = "basic_model"
    algorithm_version = OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION
    assert (
        offloading_tool.offload(model_name="basic_model") == f"Offloading [{algorithm_version}]: {model_name}"
    )


if __name__ == "__main__":
    pytest.main()
