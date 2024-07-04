class ModelManager:

    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.num_layers = 0
        self.layers_sizes = []
        self.inference_time_device = []
        self.inference_time_edge = []

        self.load_model_information()

    def load_model_information(self):
        self.num_layers = 5
        self.layers_sizes = [1000000, 1000000, 1000000, 1000000, 1000000]  # Example layer sizes
        self.inference_time_device = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example inference times on device
        self.inference_time_edge = [0.05, 0.1, 0.15, 0.2, 0.25]  # Example inference times on edge

    def get_info(self):
        return self.__dict__
