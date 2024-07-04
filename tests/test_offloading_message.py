import json

import pytest

from flask_server.offloading.offloading_message import OffloadingMessage


def test_get_message_data():
    message_data = {'payload': json.dumps({'key': 'value'})}
    message = OffloadingMessage(message_data=message_data)
    payload, message_size_bits, message_received_timestamp = message.get_message_data()

    assert payload == {'key': 'value'}
    assert isinstance(message_size_bits, int)
    assert isinstance(message_received_timestamp, float)


def test_evaluate_latency_and_speed():
    message_data = {'payload': json.dumps({'key': 'value'})}
    message = OffloadingMessage(message_data=message_data, use_synthetic_latency=False)
    latency, avg_speed = message.evaluate_latency_and_speed()

    assert isinstance(latency, float)
    assert isinstance(avg_speed, float)


def test_get_message_payload():
    message_data = {'payload': json.dumps({'key': 'value'})}
    payload = OffloadingMessage.get_message_payload(message_data)
    assert payload == {'key': 'value'}


def test_get_message_size():
    message = json.dumps({'key': 'value'})
    size_in_bits = OffloadingMessage.get_message_size(message)
    assert isinstance(size_in_bits, int)


if __name__ == "__main__":
    pytest.main()
