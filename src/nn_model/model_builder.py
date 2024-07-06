import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class ModelBuilder:
    def __init__(self, image_size: int = 10, num_classes: int = 2, image_paths: [] = list, labels: [] = list,
                 dataset_path: str = '', save_path: str = ''):
        self.image_size = image_size
        self.num_classes = num_classes
        self.image_paths = image_paths
        self.labels = labels
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.model = self.build_model()

    def build_model(self):
        logger.info("Building the model")
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 3), name="layer_0"))
        model.add(layers.MaxPooling2D((2, 2), name="layer_1"))
        model.add(layers.Flatten(name="layer_2"))
        model.add(layers.Dense(64, activation='relu', name="layer_3"))
        model.add(layers.Dense(self.num_classes, activation='sigmoid', name="layer_4"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def prepare_data(self):
        logger.info("Preparing data for the model")
        images = []
        for path in self.image_paths:
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(self.dataset_path, path),
                target_size=(self.image_size, self.image_size)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array /= 255.0
            images.append(img_array)
        x_train, x_test, y_train, y_test = train_test_split(images, self.labels, test_size=0.2, random_state=42)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def train_model(self, epochs=10):
        logger.info(f"Training the model for [{epochs}] epochs")
        x_train, x_test, y_train, y_test = self.prepare_data()
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    def evaluate_model(self):
        x_train, x_test, y_train, y_test = self.prepare_data()
        loss, accuracy = self.model.evaluate(x_test, y_test)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def predict(self, image_path: str):
        if image_path is None:
            logger.error("No data provided for prediction.")
            return None
        logger.info("Making a new prediction")
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(self.image_size, self.image_size))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = self.model.predict(img_array)
        return prediction

    def save_model_as_h5(self, model_name: str = 'test_model'):
        os.makedirs(os.path.dirname(f'models/{model_name}'), exist_ok=True)
        model_path = f'{self.save_path}/{model_name}/{model_name}.h5'
        self.model.save(model_path)
        logger.info(f"Model saved at path: {model_path}")
