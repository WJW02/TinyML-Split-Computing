import json

from logger.Logger import Logger
from offloading_tools.offloading_device import OffloadingDevicesManager
from offloading_tools.offloading_message import OffloadingMessage

logger = Logger().get_logger(__name__)


class OffloadingCommunicationHandler:
    def __init__(self):
        self.device_manager = OffloadingDevicesManager()

    def handle_incoming_message(self, message: json, device_id: str = None):
        logger.info("Reading received message")

        if device_id is None:
            logger.error("Error: Device ID cannot be None.")
            raise ValueError("Device ID cannot be None.")

        offloading_message = OffloadingMessage(message_data=message)
        offloading_information = offloading_message.get_message_offloading_info()
        device_layers_inference_time = offloading_information['layers_inference_time']
        self.device_manager.remove_outdated_devices()
        self.device_manager.update_connected_devices(device_id, offloading_message)
        self.device_manager.update_device_inference_time(device_id, device_layers_inference_time)

        return

    def get_communication_status(self) -> json:
        return self.device_manager.get_connected_devices(as_dict=True)
