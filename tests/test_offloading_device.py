import time

import pytest

from offloading_tools.offloading_device import OffloadingDevice, OffloadingDevicesManager


def test_add_message(message):
    device = OffloadingDevice(device_id="device_123")
    device.add_message(message)

    assert len(device.sent_messages) == 1


def test_get_last_message(message):
    device = OffloadingDevice(device_id="device_123")
    device.add_message(message)

    last_message = device.get_last_message()
    assert last_message == message


def test_update_connected_devices(message):
    manager = OffloadingDevicesManager()
    new_device = OffloadingDevice(device_id="device_123")
    manager.update_connected_devices("device_123", message)
    assert new_device.device_id in [dev.device_id for dev in manager.connected_devices]


def test_remove_outdated_devices(message):
    # Create an outdated message
    outdated_message = message
    outdated_message.message_received_timestamp = time.time() - 20  # Outdated message

    manager = OffloadingDevicesManager()
    device = OffloadingDevice(device_id="device_123")
    device.add_message(outdated_message)

    manager.update_connected_devices("device_123", message)

    manager.remove_outdated_devices()
    assert "device_123" not in manager.connected_devices


def test_get_device(message):
    manager = OffloadingDevicesManager()
    device = OffloadingDevice(device_id="device_123")
    manager.update_connected_devices("device_123", message)
    manager.update_connected_devices("device_1234", message)

    retrieved_device = manager.get_device("device_123")
    assert retrieved_device.device_id == device.device_id


if __name__ == "__main__":
    pytest.main()
