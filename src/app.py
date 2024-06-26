from logging import getLogger

from flask_server import init_app

if __name__ != "__main__":
    gunicorn_logger = getLogger("gunicorn.error")
else:
    gunicorn_logger = None

app = init_app(gunicorn_logger)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
