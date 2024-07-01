import json
import random
import sys
import time

from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class OffloadingMessage:
    def __init__(self, message_data: json, use_synthetic_latency: bool = True):
        self.message_data = message_data
        self.payload = None
        self.message_size_bits = None
        self.message_received_timestamp = time.time()

        # Latency and speed evaluation
        self.use_synthetic_latency = use_synthetic_latency
        self.latency, self.transfer_speed = self.evaluate_latency_and_speed()

    def get_message_data(self) -> (str, int, float):
        self.payload = self.get_message_payload(self.message_data)
        self.message_size_bits = self.get_message_size(self.message_data)
        return self.payload, self.message_size_bits, self.message_received_timestamp

    def evaluate_latency_and_speed(self) -> (float, float):
        # Calculate Latency in seconds and the average_speed
        current_time = time.time()  # time.time_ns() / 1_000_000_000
        latency = current_time - self.message_received_timestamp
        if self.use_synthetic_latency is True:
            synthetic_latency = random.uniform(1, 30)
            latency *= synthetic_latency
        avg_speed = self.message_size_bits / latency
        return self.latency, avg_speed

    @staticmethod
    def get_message_payload(message: json) -> str:
        logger.info("Extracting payload from message")
        try:
            # Ensure message.text is a string before decoding
            if isinstance(message.text, str):
                payload = message.text
            else:
                payload = message.text.decode('utf-8')
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.error("Error: Payload is not a valid JSON.")
            return None
        except AttributeError:
            logger.error("Error: Message object does not have text attribute.")
            return None

    @staticmethod
    def get_message_size(message: str) -> int:
        logger.info("Extracting size(bits) from message")
        size_in_bytes = sys.getsizeof(message)
        size_in_bits = size_in_bytes * 8
        return size_in_bits


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


class OffloadingDevicesManager:
    def __init__(self):
        self.connected_devices = {}
        self.outdated_device_threshold = 10

    def remove_outdated_devices(self) -> None:
        logger.info("Removing outdated devices")
        # Calculate time limit in seconds
        time_limit = time.time() - self.outdated_device_threshold
        # Remove outdated devices if their last message timestamp is older than the time limit
        for device in self.connected_devices:
            last_message = device.get_last_message()
            if last_message is None or last_message.message_received_timestamp < time_limit:
                logger.info(f"Removing device {device.device_id} due to outdated message")
                del self.connected_devices[device]
        logger.info(f"Connected devices ({len(self.connected_devices)}): {self.connected_devices}")

    def get_device(self, device_id: str) -> OffloadingDevice:
        for device in self.connected_devices:
            if device.device_id == device_id:
                return device
        return None

    def update_connected_devices(self, device_id: str, offloading_message: OffloadingMessage) -> None:
        logger.info("Checking connected devices")
        if device_id not in self.connected_devices.keys():
            logger.debug("Adding a new device")
            device = OffloadingDevice(device_id)
        else:
            logger.debug("Device already exists")
            device = self.get_device(device_id)
        device.add_message(offloading_message)
        logger.info(f"Device {device_id} connected")
        logger.info(f"Connected devices ({len(self.connected_devices)}): {self.connected_devices}")


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
