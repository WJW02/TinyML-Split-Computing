from flask import Flask
from flask_smorest import Api

from flask_server.configs.configs import OffloadingApiConfigs


def init_app(gunicorn_logger=None):
    # create flask app and add additional config (e.g. flask_smorest/swagger config)
    app = Flask(__name__)
    app.config.from_object(OffloadingApiConfigs)

    # use gunicorn's logger if provided
    if gunicorn_logger is not None:
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)

    # create API and Blueprint
    api = Api(app)

    # import and register handle_users blueprint
    from flask_server.offloading.views import offloading_blp
    api.register_blueprint(offloading_blp)

    return app
