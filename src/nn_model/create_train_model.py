import os

from configs.configs import NeuralNetworkModelsConfigs
from custom_model import CustomModel
from model_data import ModelData

# Configurations for model and dataset
model_name = NeuralNetworkModelsConfigs.DEFAULT_MODEL_NAME
model_dir = NeuralNetworkModelsConfigs.MODELS_DIR_PATH
image_size = NeuralNetworkModelsConfigs.IMAGE_SIZE
dataset_path = NeuralNetworkModelsConfigs.DATASET_PATH
model_store_dir = NeuralNetworkModelsConfigs.MODELS_DIR_PATH
num_samples = 100
epochs = 10
# Set a threshold for classification (you can adjust this based on your needs)
threshold = NeuralNetworkModelsConfigs.BINARY_CLASSIFICATION_THRESHOLD

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

# Build and train the model
custom_model = CustomModel(
    num_classes=1,
    save_path=model_store_dir,
    model_data=model_data
)

custom_model.build_model()
custom_model.train_model(epochs=epochs)
custom_model.evaluate_model()
custom_model.save_model(model_name=model_name)

# Simple prediction to test the model
correct_predictions = 0
for data, label in zip(model_data.images_paths, model_data.labels):
    image_path = data
    image_array = model_data.get_image_as_raw(image_path)
    prediction = custom_model.predict(image_path=None, img_array=image_array)
    # Check if the prediction is greater than the threshold
    predicted_value = 1 if prediction[0][0] >= threshold else 0
    if predicted_value == label[0]:
        correct_predictions += 1
print(f"Correct Predictions: {correct_predictions}/{len(model_data.images_paths)}")
