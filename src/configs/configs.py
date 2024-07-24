class OffloadingApiConfigs:
    API_TITLE = "Offloading Manager APIs"
    API_VERSION = "0.1.0"
    OPENAPI_VERSION = "3.1.0"
    OPENAPI_URL_PREFIX = "/api/offloading/"
    OPENAPI_SWAGGER_UI_PATH = "/docs"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    OPENAPI_REDOC_PATH = "/redoc"
    OPENAPI_REDOC_URL = "https://cdn.jsdelivr.net/npm/redoc/"


class OffloadingManagerConfigs:
    DEFAULT_WORKING_STRATEGY = "static"
    ALLOWED_WORKING_STRATEGIES = ["static", "dynamic"]
    DEFAULT_START_LAYER_INDEX = 0


class OffloadingModelConfig:
    MODEL_PATH = f"./custom_models/models/<MODEL_NAME>/test_model.keras"
    MODEL_ANALYTICS_PATH = f"./custom_models/models/<MODEL_NAME>/analytics.json"


class CustomModelExample:
    DEFAULT_MODEL_NAME: str = "test_model"
    MODELS_DIR_PATH: str = "models"
    DATASET_PATH: str = f"{MODELS_DIR_PATH}/{DEFAULT_MODEL_NAME}/data"
    MODEL_PATH: str = f"{MODELS_DIR_PATH}/{DEFAULT_MODEL_NAME}"
    IMAGE_SIZE: int = 10
    BINARY_CLASSIFICATION_THRESHOLD: float = 0.5
    NUM_TRAINING_SAMPLES: int = 100
    NUM_TRAINING_EPOCH: int = 10
