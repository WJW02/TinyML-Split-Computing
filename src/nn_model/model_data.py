import numpy as np
from PIL import Image, ImageDraw, ImageFont

from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class ModelData:
    def __init__(self, image_size: int = 10, num_images: int = 2, dataset_path: str = ''):
        self.image_size = image_size
        self.num_images = num_images
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.images_paths = []
        self.images_as_raw = []

    def generate_dataset(self):
        logger.info("Creating Data for Inference and Training")
        # Generate random images with values between 1 and 10
        rand_list = np.random.randint(0, 2, self.num_images)
        # Create labels: 1 if the image contains the number 1, else 0
        self.labels = np.array([[1] if rand_num == 1 else [0] for rand_num in rand_list])
        for i in range(len(self.labels)):
            label_text = str(self.labels[i][0])
            image_path = f'image_{i}.png'
            image = self.create_image(label_text)
            image_raw = self.get_image_as_raw(image=image)
            self.save_image(image, image_path)
            self.images.append(image)
            self.images_paths.append(image_path)
            self.images_as_raw.append(image_raw)

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

    def save_image(self, image, image_path):
        image.save(f'{self.dataset_path}/{image_path}')

    def get_image_as_raw(self, image: str or None, image_path: str or None = None):
        if image_path is not None:
            image = Image.open(image_path)  # Open the PNG image
            image = image.resize((self.image_size, self.image_size))  # Resize the image to the desired width and height
        image = image.convert('L')  # Convert the image to grayscale
        img_array = np.array(image)  # Convert the image to a NumPy array
        flat_array = img_array.flatten()  # Flatten the 2D array to a 1D array (row-major order)
        return flat_array

    def raw_image_to_png(self, image_raw):
        # Convert the raw data to a NumPy array
        flat_array = np.frombuffer(image_raw, dtype=np.uint8)
        # Reshape the 1D array to a 2D array
        img_array = flat_array.reshape((self.image_size, self.image_size))
        # Create an Image object from the NumPy array
        image = Image.fromarray(img_array)
        return image
