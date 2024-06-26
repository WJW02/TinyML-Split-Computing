from flask_server.configs.configs import OffloadingManagerConfigs


class OffloadingManager:
    def __init__(self):
        self.algorithm_version = OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION

    def offload(self, model_name: str = None) -> str:
        if model_name is None:
            raise ValueError("Error: parameter [model_name] cannot be None")
        return f"Offloading [{self.algorithm_version}]: {model_name}"
