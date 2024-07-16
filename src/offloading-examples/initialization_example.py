from custom_models.model_data import ModelData


if __name__ == "__main__":
    model_data = ModelData()
    model_data.load_from_path(path='../custom_models/models/test_model/data')
    image_path, label = (model_data.images_paths[0], model_data.labels[0])
    image_array = model_data.get_image_as_raw(image_path, expand_dims=True)
