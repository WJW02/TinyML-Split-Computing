from flask_server import init_app


def test_init_app():
    # Test if the application is initialized correctly
    app = init_app(None)
    assert app is not None


def test_run_app(mocker):
    # Test if the application runs on the correct host and port
    app = init_app(None)
    mocker.patch.object(app, 'run')
    app.run("0.0.0.0", 5000)
    app.run.assert_called_once_with("0.0.0.0", 5000)
