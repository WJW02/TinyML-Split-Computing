import json

from flask_server.offloading.offloading_device import OffloadingDevicesManager
from flask_server.offloading.offloading_message import OffloadingMessage
from logger.Logger import Logger

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
        self.device_manager.remove_outdated_devices()
        self.device_manager.update_connected_devices(device_id, offloading_message)

    def get_communication_status(self) -> json:
        return self.device_manager.get_connected_devices(as_dict=True)
