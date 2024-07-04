import json

import pytest

from flask_server.offloading.offloading_message import OffloadingMessage


class MockResponse:
    def __init__(self, text):
        self.payload = text


@pytest.fixture
def message():
    message_data = MockResponse(json.dumps({'key': 'value'}))
    return OffloadingMessage(message_data=message_data)
