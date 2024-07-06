import time

from logger.Logger import Logger
from tf_model_manager import TensorflowModelManager

logger = Logger().get_logger(__name__)


class ModelManager:

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.num_layers = 0
        self.layers_sizes = []
        self.inference_time_device = []
        self.inference_time_edge = []
        self.model_analytics = {}
        self.predictions = []
        self.model_manager = TensorflowModelManager(model_name, model_path)
        self.model_type = 'tf'

    def load_fake_model_information(self):
        self.num_layers = 5
        self.layers_sizes = [1000000, 1000000, 1000000, 1000000, 1000000]  # Example layer sizes
        self.inference_time_device = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example inference times on device
        self.inference_time_edge = [0.05, 0.1, 0.15, 0.2, 0.25]  # Example inference times on edge

    def load_model_information(self):
        return self.model_manager.num_layers

    def get_info(self):
        return self.__dict__

    def make_prediction(self, input_data, start_layer_index=0):
        self.predictions = []
        for layer_id in range(start_layer_index, self.model_manager.num_layers):
            t_begin = time.time()
            if layer_id == 0:
                layer_output = self.model_manager.predict_single_layer(layer_id, input_data)
            else:
                previous_layer_output = self.predictions[-1]

                layer_output = self.model_manager.predict_single_layer(layer_id, previous_layer_output)
            self.predictions.append(layer_output)
            t_end = time.time()

            # Updates the analytics of the model
            self.evaluate_analytics(
                layer_name=layer_id,
                layer_inference_time=t_end - t_begin,
                layer_size=self.model_manager.calculate_layer_size_in_bits(layer_output)
            )

    def evaluate_analytics(self, layer_name, layer_inference_time, layer_size):
        logger.info(f"Updating model analytics")
        self.model_analytics[layer_name] = {
            'layer_size': layer_size,
            'layer_inference_time': layer_inference_time
        }
