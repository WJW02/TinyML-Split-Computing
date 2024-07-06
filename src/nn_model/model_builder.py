import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from logger.Logger import Logger
from model_data import ModelData

logger = Logger().get_logger(__name__)


class ModelBuilder:
    def __init__(self, num_classes: int = 2, save_path: str = '', model_data: ModelData = None):
        self.model_data = model_data
        self.num_classes = num_classes
        self.save_path = save_path
        self.model = self.build_model()

    def build_model(self):
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
        return model

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
        if image_path is not None:
            logger.info("Image data will be loaded from file")
            img_array = self.model_data.get_image_as_raw(image_path)
        else:
            logger.info("Image data passed as np array")

        img_array = np.expand_dims(img_array, axis=0)

        if image_path is None:
            img_array /= 255.0

        prediction = self.model.predict(img_array)
        return prediction

    def save_model(self, model_name: str = 'test_model'):
        os.makedirs(os.path.dirname(f'{self.save_path}'), exist_ok=True)
        model_path = f'{self.save_path}/{model_name}.keras'
        self.model.save(model_path)
        logger.info(f"Model saved at path: {model_path}")
