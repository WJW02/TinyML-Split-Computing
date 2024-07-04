import time

from flask_server.offloading.offloading_message import OffloadingMessage
from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class OffloadingDevice:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.sent_messages = []

    def add_message(self, message: OffloadingMessage) -> None:
        self.sent_messages.append(message)

    def get_last_message(self) -> OffloadingMessage:
        if len(self.sent_messages) == 0:
            return None
        return self.sent_messages[-1]

    def to_dict(self) -> dict:
        return {
            'device_id': self.device_id,
            'sent_messages': [message.to_dict() for message in self.sent_messages]
        }


class OffloadingDevicesManager:
    def __init__(self):
        self.connected_devices = []
        self.outdated_device_threshold = 10000  # 10 seconds

    def remove_outdated_devices(self) -> None:
        logger.info("Removing outdated devices")
        # Calculate time limit in seconds
        time_limit = time.time() - self.outdated_device_threshold
        # Remove outdated devices if their last message timestamp is older than the time limit
        for device in self.connected_devices:
            last_message = device.get_last_message()
            if last_message is None or last_message.message_received_timestamp < time_limit:
                logger.info(f"Removing device {device.device_id} due to outdated message")
                self.connected_devices.remove(device)
        logger.info(f"Connected devices ({len(self.connected_devices)}): {self.connected_devices}")

    def get_device(self, device_id: str) -> OffloadingDevice:
        for device in self.connected_devices:
            if device.device_id == device_id:
                return device
        return None

    def update_connected_devices(self, device_id: str, offloading_message: OffloadingMessage) -> None:
        logger.info("Checking connected devices")

        if not self.get_device(device_id):
            logger.debug("Adding a new device")
            device = OffloadingDevice(device_id)
            self.connected_devices.append(device)
        else:
            logger.debug("Device already exists")
            device = self.get_device(device_id)
        device.add_message(offloading_message)
        logger.info(f"Device {device_id} connected")
        logger.info(f"Connected devices: ({len(self.connected_devices)}): {self.get_connected_devices(as_dict=True)}")

    def get_connected_devices(self, as_dict: bool = False) -> list:
        if as_dict:
            return [device.to_dict() for device in self.connected_devices]
        return self.connected_devices
