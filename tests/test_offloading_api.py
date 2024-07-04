from unittest.mock import patch

import pytest
from flask import Flask

from flask_server.offloading.views import offloading_blp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(offloading_blp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_offloading_api_success(client):
    request_body = {"device_id": "basic_device", "model_name": "basic_model", "payload": {'key': 'value'}}
    response = client.post('api/perform-offloading', json=request_body)
    assert response.status_code == 200


def test_offloading_api_failure_wrong_args(client):
    response = client.post('api/perform-offloading', json={"wrong_key": "basic_model"})
    assert response.status_code == 400


def test_offloading_api_failure_no_args(client):
    response = client.post('api/perform-offloading', json={})
    assert response.status_code == 400


@patch('flask_server.offloading.views.offloading_communication_handler.get_communication_status')
def test_get_offloading_communication_status_success(mock_get_communication_status, client):
    mock_get_communication_status.return_value = {"device": "[]"}
    response = client.get('api/get-offloading-communication-status')
    assert response.status_code == 200
    assert response.json == {"code": 200, "message": "Success", "communication_data": {"device": "[]"}}


@patch('flask_server.offloading.views.offloading_communication_handler.get_communication_status')
def test_get_offloading_communication_status_unexpected_error(mock_get_communication_status, client):
    mock_get_communication_status.side_effect = Exception("Unexpected error")
    response = client.get('api/get-offloading-communication-status')
    assert response.status_code == 500


if __name__ == "__main__":
    pytest.main()
