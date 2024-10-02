import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time
import csv 
import os

def calculate_size_in_bits(layer_output):
    # Get the data type of the layer output
    dtype = layer_output.dtype
    # Calculate the size in bits
    size_in_bits = tf.reduce_prod(layer_output.shape) * tf.constant(dtype.itemsize * 8)
    return size_in_bits.numpy()  # Convert to a Python scalar for writing to CSV

def create_analytics_csv(analytics_path, layer_name, layer_inference_time, layer_size):
    with open(analytics_path, mode='a', newline='') as csv_file:
        fieldnames = ['layer', 'layer_size', 'layer_inference_time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Check if the file is empty and write the header if needed
        if csv_file.tell() == 0:
            writer.writeheader()

        writer.writerow({
            'layer': layer_name,
            'layer_size': layer_size,
            'layer_inference_time': layer_inference_time
        })
    print(f"\n{layer_name}, {layer_size}, {layer_inference_time}")

def predict_single_layer(layer, layer_input_data, analytics_path):
    # Create an intermediate model with the current layer
    intermediate_model = tf.keras.Model(inputs=layer.input, outputs=layer.output)
    print(layer.input, layer.output)
    # Predict using the current layer, keeps track of the time it takes
    t_begin = time.time_ns() / 1_000_000_000
    layer_output = intermediate_model.predict(layer_input_data)
    t_end = time.time_ns() / 1_000_000_000

    create_analytics_csv(
        analytics_path = analytics_path,
        layer_name = layer.name,
        layer_inference_time = t_end - t_begin,
        layer_size = calculate_size_in_bits(layer_output)
    )
    
    return layer_output

def perform_predict(model, input_data, analytics_path, start_layer_index=0):
    num_layers = len(model.layers)
    predictions = {}
    
    # For sequential models start_layer_offset = 0
    # For non-sequential models start_layer_offset = 1 (because the first one is an InputLayer)
    if (isinstance(model.layers[start_layer_index], tf.keras.layers.InputLayer)):
        start_layer_offset = 1
    else:
        start_layer_offset = 0

    for layer_id in range(start_layer_index+start_layer_offset, num_layers):
        layer = model.layers[layer_id]

        # For the first layer, use the preprocessed input image
        if layer_id == start_layer_offset:
            layer_output = predict_single_layer(layer, input_data, analytics_path)
        else:
            # Get the previous layers' output tensor
            inbound_node = layer._inbound_nodes[0]
            if isinstance(inbound_node.inbound_layers, list):
                previous_layer_output = []
                for inbound_layer in inbound_node.inbound_layers:
                    previous_layer_output.append(predictions[inbound_layer])
            else:
                previous_layer_output = predictions[inbound_node.inbound_layers]

            layer_output = predict_single_layer(layer, previous_layer_output, analytics_path)
        # Save this layer's output tensor in dict to be referenced by next layers
        predictions[layer] = layer_output

    return predictions

def get_input_data(nn_id):
    # Load and preprocess the input image for the first layer
    input_image = load_img(f'./models/{nn_id}/pred_data/pred_test_is_1.png', color_mode="grayscale", target_size=(96, 96))
    input_array = img_to_array(input_image)
    input_array = np.array([input_array])
    return input_array

nn_id = 'test_model'
model = tf.keras.models.load_model(f'./models/{nn_id}/{nn_id}.h5')
input_data = get_input_data(nn_id=nn_id)
analytics_path = f'./models/{nn_id}/analytics_data/analytics.csv'
os.makedirs(analytics_path.removesuffix('analytics.csv'), exist_ok=True)
predictions = perform_predict(model=model, input_data=input_data, analytics_path=analytics_path)
prediction = list(predictions.items())[-1][0]
print(f'prediction: {prediction}')