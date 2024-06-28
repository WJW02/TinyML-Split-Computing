import pytest
from flask import Flask

from flask_server.configs.configs import OffloadingApiMessages
from flask_server.offloading.views import offloading_blp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["TEXT_CLASSIFIER_MODEL"] = "__DEFAULT__"
    app.register_blueprint(offloading_blp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_offloading_api_success(client):
    response = client.post('api/perform-offloading', json={"model_name": "basic_model"})
    assert response.status_code == 200


def test_offloading_api_failure_wrong_args(client):
    response = client.post('api/perform-offloading', json={"wrong_key": "basic_model"})
    assert response.status_code == 422


def test_offloading_api_failure_no_args(client):
    response = client.post('api/perform-offloading', json={})
    assert response.status_code == 400


if __name__ == "__main__":
    pytest.main()
