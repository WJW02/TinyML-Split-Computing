import json
import time

import pandas as pd

from custom_models.custom_model import CustomModel
from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class OffloadingModel:

    def __init__(self, model_name, model_path, model_analytics_path, load_model: bool = False,
                 load_model_data: bool = False):
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

        if load_model:
            self.load_custom_model(model_path)

        if load_model_data:
            self.load_model_analytics(model_analytics_path)

    def get_info(self):
        return self.to_dict(keep_model=False, filter_predictions=True)

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

    #TODO: FIX bug on evaluation of layers that saves only the used layer on a certain predictio
    def trigger_prediction(self, input_data, start_layer_index: int = 0, end_layer_index: int = None):
        # TODO: Remove this
        self.predictions = [] if start_layer_index == 0 else [input_data]

        logger.info(f"Triggering prediction from start_layer_index: {start_layer_index}")

        # We use only the layers from start_layer_index to end_layer_index
        end_layer_index = self.num_layers - 1 if end_layer_index is None else end_layer_index
        if end_layer_index > self.num_layers:
            logger.warning(f"Invalid end_layer_index: {end_layer_index}. Setting to {self.num_layers}")
            end_layer_index = self.num_layers
        logger.info(f"Predicting layers from {start_layer_index} to {end_layer_index}")
        layers_to_use = self.custom_model.model.layers[start_layer_index:end_layer_index]

        for layer_index, layer in enumerate(layers_to_use):
            layer_index += start_layer_index
            if layer_index == end_layer_index:
                break
            logger.info(f"Predicting Layer: {layer_index}")
            start_time = time.time()

            prediction_data = input_data if layer_index == 0 else self.predictions[-1]
            prediction = self.custom_model.predict_single_layer(layer_index, prediction_data)
            end_time = time.time()

            self.predictions.append(prediction)
            layer_inference_time = end_time - start_time
            layer_size = self.custom_model.get_layer_size_in_bits(layer_output=prediction)

            self.evaluate_analytics(
                layer_name=layer_index,
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
        logger.info("Storing model analytics")
        # Updates the attributes related to layers inference times
        self.num_layers = self.custom_model.num_layers
        self.layers_sizes, self.layers_inference_time = [], []
        for layer in self.model_analytics.keys():
            self.layers_sizes.append(self.model_analytics[layer]['layer_size'])
            self.layers_inference_time.append(self.model_analytics[layer]['layer_inference_time'])

        logger.info(f"Model Analytics: {str(self.model_analytics)}")
        model_analytics_df = pd.DataFrame.from_dict(self.model_analytics)
        model_analytics_df.to_json(self.model_analytics_path)

    def to_dict(self, keep_model: bool = False, filter_predictions: bool = False):
        offloading_model_as_dict = self.__dict__
        offloading_model_as_dict['custom_model'] = self.custom_model.__dict__
        # Remove the custom model from the dict
        if not keep_model:
            del offloading_model_as_dict['custom_model']['model']
        if filter_predictions:
            if self.predictions:
                offloading_model_as_dict['predictions'] = offloading_model_as_dict['predictions'][-1]
        return offloading_model_as_dict

    def perform_model_initialization(self, input_data):
        logger.info("Performing Model Initialization")
        self.trigger_prediction(input_data=input_data, start_layer_index=0)
        return self.get_info()
