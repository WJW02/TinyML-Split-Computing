import json

import requests

from logger.Logger import Logger

logger = Logger(config_path="../logger/logger_config.json").get_logger(__name__)


class SimpleDevice:
    def __init__(self, device_id: str, message_data: dict):
        self.device_id = device_id
        self.message_data = message_data
        self.response = None

    def send_message(self) -> None:
        self.message_data['device_id'] = self.device_id
        logger.info(f"Sending message: {json.dumps(self.message_data)}")
        self.make_offloading_request(self.message_data)

    def make_offloading_request(self, message_data) -> None:
        try:
            self.response = requests.post("http://localhost:8080/api/perform-offloading", json=message_data)
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    message_data_dict = {
        "model_name": "model_name",
        "start_layer_index": 0,
        "working_strategy": "working_strategy",
        "algorithm_version": "algorithm_version",
        "payload": {"message": "payload_string"}
    }

    device = SimpleDevice(device_id="test2", message_data=message_data_dict)
    device.send_message()
    print(device.response.text)
