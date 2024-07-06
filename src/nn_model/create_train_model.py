import os

from model_data import ModelData
# from model_builder import ModelBuilder
from flask_server.configs.configs import NeuralNetworkModelsConfigs

model_name = NeuralNetworkModelsConfigs.DEFAULT_MODEL_NAME
model_dir = NeuralNetworkModelsConfigs.MODELS_DIR_PATH
image_size = NeuralNetworkModelsConfigs.IMAGE_SIZE
dataset_path = NeuralNetworkModelsConfigs.DATASET_PATH
num_samples = 5

os.makedirs(f'{model_dir}/{model_name}/data', exist_ok=True)
os.makedirs(f'{model_dir}/{model_name}/analytics', exist_ok=True)

# Create an instance of ModelDataGenerator
model_data = ModelData(
    image_size=image_size,
    num_images=num_samples,
    dataset_path=dataset_path
)
model_data.generate_dataset()