import inspect

from flask_server.configs.configs import OffloadingManagerConfigs


class OffloadingManager:
    def __init__(self,
                 model_name: str = None,
                 algorithm_version: str = None,
                 working_strategy: str = None,
                 start_layer_index: int = None
                 ):

        # Set up allowed working strategies
        self.allowed_working_strategies = OffloadingManagerConfigs.ALLOWED_WORKING_STRATEGIES

        # Set up with provided or default values
        self.model_name = model_name or OffloadingManagerConfigs.DEFAULT_MODEL_NAME
        self.algorithm_version = algorithm_version or OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION
        self.working_strategy = working_strategy or OffloadingManagerConfigs.DEFAULT_WORKING_STRATEGY
        self.start_layer_index = start_layer_index or OffloadingManagerConfigs.DEFAULT_START_LAYER_INDEX

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value is None:
            raise ValueError("Error: parameter [model_name] cannot be None")
        self._model_name = value

    @property
    def algorithm_version(self):
        return self._algorithm_version

    @algorithm_version.setter
    def algorithm_version(self, value):
        method_name = inspect.currentframe().f_back.f_code.co_name
        if value is None:
            raise ValueError(f"Error: parameter [{method_name}] cannot be None")
        self._algorithm_version = value

    @property
    def working_strategy(self):
        return self._working_strategy

    @working_strategy.setter
    def working_strategy(self, value):
        if value is None or value not in self.allowed_working_strategies:
            raise ValueError(f"Error: The given [working_strategy] is not valid. "
                             f"Allowed strategies are: {self.allowed_working_strategies}")
        self._working_strategy = value

    @property
    def start_layer_index(self):
        return self._start_layer_index

    @start_layer_index.setter
    def start_layer_index(self, value):
        if value is None or not isinstance(value, int) or value < 0:
            raise ValueError("Error: parameter [start_layer_index] must be a non-negative integer")
        self._start_layer_index = value

    def offload(self) -> str:
        return f"Offloading [{self.algorithm_version}]: {self.model_name}"
