import json
import random
import sys
import time

from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class OffloadingMessage:
    def __init__(self, message_data: str, use_synthetic_latency: bool = True):
        self.message_data = message_data
        self.payload = self.get_message_payload(self.message_data)
        self.message_size_bits = self.get_message_size(self.message_data)
        self.message_received_timestamp = time.time_ns() / 1_000_000_000

        # Latency and speed evaluation
        self.use_synthetic_latency = use_synthetic_latency
        self.latency, self.transfer_speed = self.evaluate_latency_and_speed()

    def get_message_data(self) -> (dict, int, float):
        return self.payload, self.message_size_bits, self.message_received_timestamp

    def evaluate_latency_and_speed(self) -> (float, float):
        # Calculate Latency in seconds and the average_speed
        current_time = time.time_ns() / 1_000_000_000
        latency = float(float(current_time) - float(self.message_received_timestamp))
        if self.use_synthetic_latency is True:
            synthetic_latency = random.uniform(1, 30)
            latency *= synthetic_latency
        avg_speed = self.message_size_bits / latency if latency != 0 else self.message_size_bits
        avg_speed = round(float(avg_speed), 3)
        return latency, avg_speed

    @staticmethod
    def get_message_payload(message: dict) -> dict:
        logger.info("Extracting payload from message")
        try:
            payload = message.get('payload')
            logger.info(f"Message payload: {payload}")
            if isinstance(payload, str):
                return {"payload": payload.encode('utf-8').decode('utf-8')}
            else:
                logger.error("Error: Message payload is not a string.")
                return None
        except json.JSONDecodeError:
            logger.error("Error: Payload is not a valid JSON.")
            return None
        except AttributeError:
            logger.error("Error: Message object does not have text attribute.")
            return None

    @staticmethod
    def get_message_size(payload_str: str) -> int:
        logger.info("Extracting size(bits) from message")
        size_in_bytes = sys.getsizeof(payload_str)
        size_in_bits = size_in_bytes * 8
        return size_in_bits

    def to_dict(self) -> dict:
        return self.__dict__
