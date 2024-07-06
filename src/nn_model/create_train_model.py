import os

# from model_builder import ModelBuilder
from configs.configs import NeuralNetworkModelsConfigs
from model_data import ModelData

# Configurations for model and dataset
model_name = NeuralNetworkModelsConfigs.DEFAULT_MODEL_NAME
model_dir = NeuralNetworkModelsConfigs.MODELS_DIR_PATH
image_size = NeuralNetworkModelsConfigs.IMAGE_SIZE
dataset_path = NeuralNetworkModelsConfigs.DATASET_PATH
num_samples = 5

# Ensure that the folders needed to store models data are created
os.makedirs(f'{model_dir}/{model_name}/data', exist_ok=True)
os.makedirs(f'{model_dir}/{model_name}/analytics', exist_ok=True)

# Create an instance of ModelDataGenerator
model_data = ModelData(
    image_size=image_size,
    num_images=num_samples,
    dataset_path=dataset_path
)
model_data.generate_dataset()
