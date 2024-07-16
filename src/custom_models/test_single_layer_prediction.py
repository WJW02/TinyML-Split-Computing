import tensorflow as tf

from configs.configs import NeuralNetworkModelsConfigs
from custom_models.custom_model import CustomModel
from custom_models.model_data import ModelData

# Configurations for model and dataset
model_name = NeuralNetworkModelsConfigs.DEFAULT_MODEL_NAME
model_dir = NeuralNetworkModelsConfigs.MODELS_DIR_PATH
image_size = NeuralNetworkModelsConfigs.IMAGE_SIZE
dataset_path = NeuralNetworkModelsConfigs.DATASET_PATH
model_path = NeuralNetworkModelsConfigs.MODELS_DIR_PATH + "/" + NeuralNetworkModelsConfigs.DEFAULT_MODEL_NAME + ".keras"
num_samples = 100
epochs = 10
threshold = NeuralNetworkModelsConfigs.BINARY_CLASSIFICATION_THRESHOLD

# Create an instance of ModelDataGenerator
model_data = ModelData(
    image_size=image_size,
    num_images=num_samples,
    dataset_path=dataset_path
)
model_data.load_from_path(dataset_path)
# Build and train the model
custom_model = CustomModel(
    model_data=model_data
)

custom_model.load_model(model_path=model_path)

# Simple prediction to test the model
correct_predictions = 0
for data, label in zip(model_data.images_paths, model_data.labels):
    image_path = data
    image_array = model_data.get_image_as_raw(image_path, expand_dims=True)
    predictions = []
    for layer_id, layer in enumerate(custom_model.model.layers):
        expected_input_shape = layer.input.shape[1:]
        prediction_data = image_array if layer_id == 0 else predictions[-1]
        if prediction_data.shape[1:] != expected_input_shape:
            prediction_data = tf.image.resize(prediction_data, expected_input_shape)
        prediction = custom_model.predict_single_layer(layer_id, prediction_data)
        predictions.append(prediction)
    print("END")
    break

for data, label in zip(model_data.images_paths, model_data.labels):
    image_path = data
    image_array = model_data.get_image_as_raw(image_path)
    prediction = custom_model.predict(image_path=None, img_array=image_array)
    # Check if the prediction is greater than the threshold
    predicted_value = 1 if prediction[0][0] >= threshold else 0
    print("END")
    break
