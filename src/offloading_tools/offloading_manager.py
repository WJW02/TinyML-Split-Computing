from configs.configs import OffloadingManagerConfigs
from logger.Logger import Logger
from offloading_tools.offloading_algo import OffloadingAlgo
from offloading_tools.offloading_device import OffloadingDevice
from offloading_tools.offloading_message import OffloadingMessage
from offloading_tools.offloading_model import OffloadingModel

logger = Logger().get_logger(__name__)


class OffloadingManager:
    def __init__(self, start_layer_index: int = None):
        self.start_layer_index = start_layer_index or OffloadingManagerConfigs.DEFAULT_START_LAYER_INDEX

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

    def offload(self, offloading_message: OffloadingMessage, model: OffloadingModel, device: OffloadingDevice) -> int:
        logger.info("Starting offloading process")
        logger.info(f"Computing Offloading: ")
        logger.info(f"Offloading Message: {offloading_message.get_message_offloading_info()}")
        logger.info(f"Device inference time: {device.layers_inference_time}")
        logger.info(f"Edge inference time: {model.layers_inference_time}")

        offloading_algo = OffloadingAlgo(
            avg_speed=offloading_message.transfer_speed,
            num_layers=model.num_layers,
            layers_sizes=model.layers_sizes,
            inference_time_device=device.layers_inference_time,
            inference_time_edge=model.layers_inference_time
        )
        best_offloading_layer = offloading_algo.static_offloading()

        result = {
            "best_offloading_layer": best_offloading_layer,
            "additional_info": {
                "offloaded_model_info": model.get_info(),
                "offloading_algo_info": offloading_algo.get_info(),
            },
        }
        logger.info(f"Offloading result: \n{result}")
        return result
