class OffloadingApiConfigs:
    API_TITLE = "Offloading Manager APIs"
    API_VERSION = "0.1.0"
    OPENAPI_VERSION = "3.1.0"
    OPENAPI_URL_PREFIX = "/api/"
    OPENAPI_SWAGGER_UI_PATH = "/docs"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    OPENAPI_REDOC_PATH = "/redoc"
    OPENAPI_REDOC_URL = "https://cdn.jsdelivr.net/npm/redoc/"


class OffloadingApiMessages:
    WRONG_NAME_KEY = "OPS: What's your name?"
    UNEXPECTED_ERROR = "ERROR: Unexpected error occurred!"
    NAME_KEY_MISSING = "OPS: You said something?"
    SUCCESS_RESPONSE = "SUCCESS: Hello, <your_name> World!"


class OffloadingManagerConfigs:
    DEFAULT_ALGORITHM_VERSION = "0.0.1"
