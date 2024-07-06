import time

import tensorflow as tf

from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class TensorflowModelManager:

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.num_layers = 0
        self.layers_sizes = []
        self.inference_time_device = []
        self.inference_time_edge = []
        self.model = None

    def load_model(self):
        self.num_layers = len(self.model.layers)
        return tf.keras.models.load_model(self.model_path)

    def get_model_layer(self, layer_id):
        return self.model.layers[layer_id]

    def get_info(self):
        return self.__dict__

    @staticmethod
    def calculate_layer_size_in_bits(layer_output):
        dtype = layer_output.dtype  # Get the data type of the layer output
        # Calculate the size in bits and convert to Python scalar for writing to CSV
        size_in_bits = tf.reduce_prod(layer_output.shape) * tf.constant(dtype.itemsize * 8)
        size_in_bits = size_in_bits.numpy()
        return size_in_bits

    def predict_single_layer(self, layer_id, layer_input_data):
        logger.info(f"Making a prediction for layer [{layer_id}]")
        layer = self.get_model_layer(layer_id)
        # Create an intermediate model with the current layer
        intermediate_model = tf.keras.Model(inputs=layer.input, outputs=layer.output)
        layer_output = intermediate_model.predict(layer_input_data)
        return layer_output



