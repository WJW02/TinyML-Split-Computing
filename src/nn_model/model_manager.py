class ModelManager:

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.num_layers = 0
        self.layers_sizes = []
        self.inference_time_device = []
        self.inference_time_edge = []
        self.load_model_information()
        self.model = None

    def load_model(self):
        self.num_layers = len(self.model.layers)
        # TODO: if the stats of the model do not exists create them so: first inference for each layer, store
        return tf.keras.models.load_model(self.model_path)

    def get_model_layer(self, layer_id):
        return self.model.layers[layer_id]

    def load_model_information(self):
        self.num_layers = 5
        self.layers_sizes = [1000000, 1000000, 1000000, 1000000, 1000000]  # Example layer sizes
        self.inference_time_device = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example inference times on device
        self.inference_time_edge = [0.05, 0.1, 0.15, 0.2, 0.25]  # Example inference times on edge

    def get_info(self):
        return self.__dict__

    @staticmethod
    def calculate_layer_size_in_bits(layer_output):
        dtype = layer_output.dtype  # Get the data type of the layer output
        # Calculate the size in bits and convert to Python scalar for writing to CSV
        size_in_bits = tf.reduce_prod(layer_output.shape) * tf.constant(dtype.itemsize * 8)
        size_in_bits = size_in_bits.numpy()
        return size_in_bits
