import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from custom_models.model_data import ModelData
from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class CustomModel:
    def __init__(self, num_classes: int = 2, save_path: str = '', model_data: ModelData = None):
        self.model_data = model_data
        self.num_classes = num_classes
        self.save_path = save_path
        self.model_path = None
        self.num_layers = None
        self.model = None
        self.model_name = None

    def build_model(self):
        logger.info(f"Building the model for image size{self.model_data.image_size}")
        logger.info("Building the model")
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(self.model_data.image_size, self.model_data.image_size, 3),
                                name="layer_0"))
        model.add(layers.MaxPooling2D((2, 2), name="layer_1"))
        model.add(layers.Flatten(name="layer_2"))
        model.add(layers.Dense(64, activation='relu', name="layer_3"))
        model.add(layers.Dense(self.num_classes, activation='sigmoid', name="layer_4"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def prepare_data(self):
        logger.info("Preparing data for the model")
        images = [self.model_data.get_image_as_raw(path) for path in self.model_data.images_paths]
        x_train, x_test, y_train, y_test = train_test_split(images, self.model_data.labels, test_size=0.2,
                                                            random_state=42)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def train_model(self, epochs=10):
        logger.info(f"Training the model for [{epochs}] epochs")
        x_train, x_test, y_train, y_test = self.prepare_data()
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    def evaluate_model(self):
        x_train, x_test, y_train, y_test = self.prepare_data()
        loss, accuracy = self.model.evaluate(x_test, y_test)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def predict(self, image_path: str, img_array):
        logger.info("Making a new prediction")
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        return prediction

    def save_model(self, model_name: str = 'test_model'):
        self.model_name = model_name
        logger.info(f"Saving the model with name: {model_name}")
        os.makedirs(f'{self.save_path}', exist_ok=True)
        model_path = f'{self.save_path}/{model_name}/{model_name}.keras'
        self.model.save(model_path)
        logger.info(f"Model saved at path: {model_path}")

    def load_model(self, model_path: str):
        logger.info(f"Loading model from path: {model_path}")
        try:
            self.model_path = model_path
            self.model = tf.keras.models.load_model(model_path)
            self.num_layers = len(self.model.layers)
        except Exception as e:
            print(f"Error loading model: {e}")
            logger.error(f"Failed to load model: {e}")

    def get_model_layer(self, layer_id):
        return self.model.layers[layer_id]

    def get_info(self):
        return self.__dict__

    @staticmethod
    def get_layer_size_in_bits(layer_output):
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

    def save_layer(self, layer_id: int, layer_name: str = None, save_tflite: bool = False):
        if layer_id >= len(self.model.layers):
            raise IndexError("Layer index out of range")

        logger.info(f"Saving layer [{layer_id}]")
        layer = self.get_model_layer(layer_id)
        intermediate_model = tf.keras.Model(inputs=layer.input, outputs=layer.output)

        # Save the layer model
        if not layer_name:
            layer_name = f"layer_{layer_id}"
        os.makedirs(f'{self.save_path}/{self.model_name}/layers/keras', exist_ok=True)
        layer_model_path = os.path.join(self.save_path, f'{self.model_name}/layers/keras/{layer_name}.keras')
        intermediate_model.save(layer_model_path)
        logger.info(f"Layer model saved at path: {layer_model_path}")

        if save_tflite:
            # Convert the model to TFLite format
            converter = tf.lite.TFLiteConverter.from_keras_model(layer)
            tflite_model = converter.convert()

            # Define the output .tflite file path
            os.makedirs(f'{self.save_path}/{self.model_name}/layers/tflite', exist_ok=True)
            tflite_file_name = layer_model_path.replace('keras', 'tflite')
            # Save the .tflite model to the specified path
            with open(tflite_file_name, 'wb') as f:
                f.write(tflite_model)

    def load_layer(self, layer_name: str):
        logger.info(f"Loading layer model from path: {layer_name}")
        try:
            layer_model_path = os.path.join(self.save_path, f'{self.model_name}/{layer_name}.keras')
            layer_model = tf.keras.models.load_model(layer_model_path)
            return layer_model
        except Exception as e:
            print(f"Error loading layer model: {e}")
            logger.error(f"Failed to load layer model: {e}")


