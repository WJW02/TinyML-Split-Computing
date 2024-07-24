import base64
import os

import requests
from configs.configs import CustomModelExample
from custom_models.model_data import ModelData

if __name__ == '__main__':
    # Loading example data
    model_data = ModelData(image_size=CustomModelExample.IMAGE_SIZE)
    dataset_path = '../custom_models/models/test_model/data'
    model_data.load_from_path(path=dataset_path)
    image_path, label = (model_data.images_paths[0], model_data.labels[0])
    image_path = os.path.join(dataset_path, image_path)
    image_array = model_data.get_image_as_raw(image_path, expand_dims=True)

    # Convert the TensorFlow EagerTensor to a NumPy array
    image_array_np = image_array.numpy()
    # Convert the image_array (tensor) to a base64-encoded string
    image_array_b64 = base64.b64encode(image_array_np.tobytes()).decode('utf-8')

    # Sending example image to the server
    url = 'http://localhost:8080/api/offloading/model-initialization'
    params = {
        'model_name': 'test_model',
        'model_data': image_array_b64,
        'shape': image_array_np.shape  # Send the shape to reconstruct the tensor on the server
    }

    response = requests.post(url, json=params)
    print(response.text)
