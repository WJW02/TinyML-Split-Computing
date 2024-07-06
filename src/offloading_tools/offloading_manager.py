import inspect

from configs.configs import OffloadingManagerConfigs
from logger.Logger import Logger
from nn_model.model_manager import ModelManager
from offloading_tools.offloading_algo import OffloadingAlgo
from offloading_tools.offloading_message import OffloadingMessage

logger = Logger().get_logger(__name__)


class OffloadingManager:
    def __init__(self,
                 algorithm_version: str = None,
                 working_strategy: str = None,
                 start_layer_index: int = None
                 ):

        # Set up allowed working strategies
        self.allowed_working_strategies = OffloadingManagerConfigs.ALLOWED_WORKING_STRATEGIES

        # Set up with provided or default values
        self.algorithm_version = algorithm_version or OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION
        self.working_strategy = working_strategy or OffloadingManagerConfigs.DEFAULT_WORKING_STRATEGY
        self.start_layer_index = start_layer_index or OffloadingManagerConfigs.DEFAULT_START_LAYER_INDEX

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

    def prepare_offloading_data(self):
        pass

    def offload(self, offloading_message: OffloadingMessage, model: ModelManager) -> int:
        logger.info("Starting offloading process")

        offloading_algo = OffloadingAlgo(
            avg_speed=offloading_message.transfer_speed,
            num_layers=model.num_layers,
            layers_sizes=model.layers_sizes,
            inference_time_device=model.inference_time_device,
            inference_time_edge=model.inference_time_edge
        )

        logger.info(f"Computing Offloading: ")
        best_offloading_layer = offloading_algo.static_offloading()

        result = {
            "best_offloading_layer": best_offloading_layer,
            "offloaded_model_info": model.get_info(),
            "offloading_algo_info": offloading_algo.get_info(),
            "additional_info": self.__dict__

        }
        logger.info(f"Offloading result: \n{result}")
        return result
