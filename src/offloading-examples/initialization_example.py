import base64
import os

import numpy as np
import requests
from PIL import Image

from custom_models.model_data import ModelData
import requests
import json
import os
import numpy as np
import base64
import tensorflow as tf

if __name__ == '__main__':
    # Loading example data
    model_data = ModelData()
    dataset_path = '../custom_models/models/test_model/data'
    model_data.load_from_path(path=dataset_path)
    image_path, label = (model_data.images_paths[0], model_data.labels[0])
    image_path = os.path.join(dataset_path, image_path)
    image_array = model_data.get_image_as_raw(image_path, expand_dims=True)

    # Convert the image_array to TensorFlow EagerTensor
    image_array_tf = tf.convert_to_tensor(image_array)

    # Convert the TensorFlow EagerTensor to a NumPy array
    image_array_np = image_array_tf.numpy()

    # Resize or crop the image to match (20, 20)
    resized_image = tf.image.resize(image_array_np, [20, 20])  # Resize using TensorFlow

    # Convert the resized image to NumPy array and normalize if necessary
    image_array_resized = resized_image.numpy()
    image_array_resized = image_array_resized / 255.0  # Normalize to [0, 1] if required

    # Convert the image_array (tensor) to a base64-encoded string
    image_array_b64 = base64.b64encode(image_array_resized.tobytes()).decode('utf-8')

    # Sending example image to the server
    url = 'http://localhost:8080/api/offloading/model-initialization'
    params = {
        'model_name': 'test_model',
        'model_data': image_array_b64,
        'shape': image_array_resized.shape  # Send the shape to reconstruct the tensor on the server
    }

    response = requests.post(url, json=params)
    print(response.text)
