import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class ModelData:
    def __init__(self, image_size: int, num_images: int = 2, dataset_path: str = ''):
        self.image_size = image_size
        self.num_images = num_images
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.images_paths = []

    def generate_dataset(self):
        logger.info("Creating Data for Inference and Training")
        logger.info(f"Images size: {self.image_size}")
        # Generate random images with values between 1 and 10
        rand_list = np.random.randint(0, 2, self.num_images)
        # Create labels: 1 if the image contains the number 1, else 0
        self.labels = np.array([[1] if rand_num == 1 else [0] for rand_num in rand_list])
        for i in range(len(self.labels)):
            label_text = str(self.labels[i][0])
            image_path = f'image_{i}.png'
            image = self.create_image(label_text)
            self.save_image_and_label(image, label_text, image_path)
            self.images.append(image)
            self.images_paths.append(image_path)

    def load_from_path(self, path: str):
        self.images_paths = os.listdir(path)
        self.images_paths.remove('labels.txt')
        with open(f'{path}/labels.txt', 'r') as f:
            self.labels = [label.strip() for label in f.readlines()]
        for image_path in self.images_paths:
            image = Image.open(os.path.join(path, image_path))
            self.images.append(image)

    def create_image(self, text: str):
        # Create a blank image
        image = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(image)
        # Use a basic font (you can customize the font if needed)
        font = ImageFont.load_default()
        # Get the text size bounding box
        text_bbox = draw.textbbox((0, 0), text, font)
        # Center the text on the image
        x = (image.width - text_bbox[2]) / 2
        y = (image.height - text_bbox[3]) / 2
        # Draw the text on the image
        draw.text((x, y), text, fill='black', font=font)
        return image

    def save_image_and_label(self, image, label_text, image_path):
        image.save(f'{self.dataset_path}/{image_path}')
        with open(f'{self.dataset_path}/labels.txt', 'a') as f:
            f.write(f"{label_text}\n")

    def get_image_as_raw(self, image_path: str, expand_dims: bool = False):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(self.dataset_path, image_path),
            target_size=(self.image_size, self.image_size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array /= 255.0  # Normalize pixel values to be between 0 and 1
        img_array = tf.image.resize(img_array, (self.image_size, self.image_size))

        if expand_dims:
            img_array = tf.expand_dims(img_array, 0)
        return img_array

    def raw_image_to_png(self, image_raw):
        # Convert the raw data to a NumPy array
        flat_array = np.frombuffer(image_raw, dtype=np.uint8)
        # Reshape the 1D array to a 2D array
        img_array = flat_array.reshape((self.image_size, self.image_size))
        # Create an Image object from the NumPy array
        image = Image.fromarray(img_array)
        return image
