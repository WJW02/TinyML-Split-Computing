import os

from src.configs.configs import CustomModelExample
from custom_model import CustomModel
from model_data import ModelData

if __name__ == '__main__':
    # Configurations for model and dataset
    model_name = CustomModelExample.DEFAULT_MODEL_NAME
    model_path = CustomModelExample.MODEL_PATH
    image_size = CustomModelExample.IMAGE_SIZE
    dataset_path = CustomModelExample.DATASET_PATH
    model_store_dir = CustomModelExample.MODELS_DIR_PATH
    num_samples = CustomModelExample.NUM_TRAINING_SAMPLES
    epochs = CustomModelExample.NUM_TRAINING_EPOCH
    # Set a threshold for classification (you can adjust this based on your needs)
    threshold = CustomModelExample.BINARY_CLASSIFICATION_THRESHOLD

    # Ensure that the folders needed to store models data are created
    os.makedirs(f'{dataset_path}', exist_ok=True)
    os.makedirs(f'{model_path}', exist_ok=True)

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

    # Save all layers
    num_layers = len(custom_model.model.layers)
    for i in range(num_layers):
        custom_model.save_layer(layer_id=i, layer_name=f'layer_{i}', save_tflite=True)