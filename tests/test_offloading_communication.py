import pytest

from offloading_tools.offloading_communication import OffloadingCommunicationHandler


def test_handle_incoming_message(message):
    handler = OffloadingCommunicationHandler()
    message_data = message.message_data
    device_id = "device_123"

    handler.handle_incoming_message(message_data, device_id)
    assert device_id in [device.device_id for device in handler.device_manager.connected_devices]


def test_handle_incoming_message_no_device_id(message):
    handler = OffloadingCommunicationHandler()
    message_data = message.message_data

    with pytest.raises(ValueError):
        handler.handle_incoming_message(message_data, None)


def test_get_communication_status(message):
    handler = OffloadingCommunicationHandler()
    message_data = message.message_data
    device_id = "device_123"
    handler.handle_incoming_message(message=message_data, device_id=device_id)

    status = handler.get_communication_status()
    assert status[0]['device_id'] == device_id
    assert status[0]['sent_messages'][0]['message_size_bits'] == message.message_size_bits

if __name__ == "__main__":
    pytest.main()
