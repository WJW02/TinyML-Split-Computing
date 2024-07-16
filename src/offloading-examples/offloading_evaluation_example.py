from SimpleDevice import SimpleDevice

if __name__ == "__main__":
    message_data_dict = {
        "model_name": "test_model",
        "start_layer_index": 0,
        "working_strategy": "working_strategy",
        "offloading_information": {
            "layers_inference_time": {
                'layer_0': 1.0,
                'layer_1': 1.8,
                'layer_2': 1.0,
                'layer_3': 1.8,
                'layer_4': 1.8,
                'layer_5': 9999999999.8,
            },
        },
        "payload": {
            "message": "payload_string",
            "input_data": "",
        }
    }

    device = SimpleDevice(device_id="test2", message_data=message_data_dict)
    device.send_message()
    print(device.response.text)
