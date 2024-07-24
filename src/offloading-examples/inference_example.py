import base64
import os

import requests
import tensorflow as tf

from configs.configs import CustomModelExample
from custom_models.custom_model import CustomModel
from custom_models.model_data import ModelData
from offloading_tools.offloading_model import OffloadingModel

if __name__ == '__main__':
    # Configurations for model and dataset
    model_name = "test_model.keras"
    model_path = "../custom_models/models/test_model/test_model.keras"
    image_size = CustomModelExample.IMAGE_SIZE

    model_data = ModelData(image_size=image_size)
    dataset_path = '../custom_models/models/test_model/data'
    model_data.load_from_path(path=dataset_path)
    image_path, label = (model_data.images_paths[0], model_data.labels[0])
    image_path = os.path.join(dataset_path, image_path)
    image_array = model_data.get_image_as_raw(image_path, expand_dims=True)

    # Build and train the model
    custom_model = CustomModel(model_data=model_data)
    custom_model.load_model(model_path=model_path)

    offloading_model = OffloadingModel(
        model_name=model_name,
        model_path=model_path,
        model_analytics_path="../custom_models/models/test_model/analytics.json",
        load_model=True,
        load_model_data=True,
    )
    offloading_model.trigger_prediction(input_data=image_array, start_layer_index=0, end_layer_index=2)
    last_prediction = offloading_model.predictions[-1]

    intermediate_data = last_prediction
    print(type(intermediate_data))
    original_shape = intermediate_data.shape
    intermediate_data_b64 = base64.b64encode(intermediate_data.tobytes()).decode('utf-8')

    # Sending example image to the server
    url = 'http://localhost:8080/api/offloading/model-inference'
    params = {
        'model_name': 'test_model',
        'model_data': intermediate_data_b64,
        'shape': original_shape,
        'start_layer_index': 2,
    }
    response = requests.post(url, json=params)
    print(response.text)
