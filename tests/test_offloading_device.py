from flask_server.offloading.offloading_device import OffloadingDevice
from flask_server.offloading.offloading_message import OffloadingMessage


def test_add_message():
    device = OffloadingDevice(device_id="device_123")
    message_data = {'text': '{"key": "value"}'}
    message = OffloadingMessage(message_data=message_data)
    device.add_message(message)

    assert len(device.sent_messages) == 1


def test_get_last_message():
    device = OffloadingDevice(device_id="device_123")
    message_data = {'text': '{"key": "value"}'}
    message = OffloadingMessage(message_data=message_data)
    device.add_message(message)

    last_message = device.get_last_message()
    assert last_message == message


if __name__ == '__main__':
    test_add_message()
    test_get_last_message()