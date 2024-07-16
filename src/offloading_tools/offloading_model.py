import json
import time

from custom_models.custom_model import CustomModel
from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class OffloadingModel:

    def __init__(self, model_name, model_path, model_analytics_path, load_model_data: bool = False):
        self.model_name = model_name
        self.model_path = model_path
        self.model_analytics_path = model_analytics_path
        self.load_model_data = load_model_data
        self.custom_model = None
        self.num_layers = 0
        self.predictions = []
        self.layers_sizes = []
        self.layers_inference_time = []
        self.model_analytics = {}

        if load_model_data:
            self.load_custom_model(model_path)
            self.load_model_analytics(model_analytics_path)

    def get_info(self):
        return self.to_dict(keep_model=False)

    def load_custom_model(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
        logger.info("Loading Custom Model")
        self.custom_model = CustomModel(self.model_name, self.model_path)
        self.custom_model.load_model(self.model_path)
        self.num_layers = self.custom_model.num_layers

    def load_model_analytics(self, model_analytics_path: str = None):
        logger.info("Loading Model Analytics")
        if model_analytics_path is not None:
            self.model_analytics_path = model_analytics_path
        logger.info(f"Model Analytics Path: {self.model_analytics_path}")
        try:
            with open(self.model_analytics_path, 'r') as f:
                self.model_analytics = json.load(f)
            self.store_model_analytics()
            logger.debug(f"Model Analytics: {self.model_analytics}")

        except Exception as e:
            logger.error(f"Failed to load model analytics: {e}")

    def trigger_prediction(self, input_data, start_layer_index: int = 0):
        self.predictions = []
        layer = self.custom_model.get_model_layer(start_layer_index)
        expected_input_shape = layer.input.shape[1:]

        for layer_id, layer in enumerate(self.custom_model.model.layers):
            start_time = time.time()
            prediction_data = input_data if layer_id == 0 else self.predictions[-1]
            prediction_data = self.custom_model.reshape_input_data(prediction_data, expected_input_shape)
            prediction = self.custom_model.predict_single_layer(layer_id, prediction_data)
            expected_input_shape = layer.input.shape[1:]
            end_time = time.time()

            self.predictions.append(prediction)
            layer_inference_time = end_time - start_time
            layer_size = self.custom_model.get_layer_size_in_bits(layer_output=prediction)

            self.evaluate_analytics(
                layer_name=layer_id,
                layer_inference_time=layer_inference_time,
                layer_size=layer_size
            )

    def evaluate_analytics(self, layer_name, layer_inference_time, layer_size):
        logger.info(f"Updating model analytics for layer: {layer_name}")
        self.model_analytics[layer_name] = {
            'layer_size': layer_size,
            'layer_inference_time': layer_inference_time
        }
        self.store_model_analytics()

    def store_model_analytics(self):
        # Updates the attributes related to layers inference times
        self.num_layers = self.custom_model.num_layers
        for layer in self.model_analytics.keys():
            self.layers_sizes.append(self.model_analytics[layer]['layer_size'])
            self.layers_inference_time.append(self.model_analytics[layer]['layer_inference_time'])

    def to_dict(self, keep_model: bool = False):
        offloading_model_as_dict = self.__dict__
        offloading_model_as_dict['custom_model'] = self.custom_model.__dict__
        # Remove the custom model from the dict
        if not keep_model:
            del offloading_model_as_dict['custom_model']['model']
        return offloading_model_as_dict

    def perform_model_initialization(self, input_data):
        logger.info("Performing Model Initialization")
        self.trigger_prediction(input_data=input_data, start_layer_index=0)
