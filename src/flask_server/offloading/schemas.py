from marshmallow import Schema, fields


class OffloadingInformationSchema(Schema):
    layers_inference_time = fields.Dict(
        keys=fields.String,
        values=fields.Float,
        required=True,
        metadata={
            'example': {
                'layer_0': 10.0,
                'layer_1': 23.8,
                'layer_2': 10.0,
                'layer_3': 23.8,
            }
        }
    )


class OffloadingEvaluationSchema(Schema):
    device_id = fields.String(required=True, metadata={'example': 'device-1', 'type': 'string'})
    model_name = fields.String(required=True)
    start_layer_index = fields.Integer(required=True)
    offloading_information = fields.Nested(OffloadingInformationSchema, required=True)


class OffloadingEvaluationMessages:
    WRONG_KEY = "OPS: Wrong API key. Try again."
    UNEXPECTED_ERROR = "ERROR: Unexpected error occurred!"
    MISSING_KEYS = "OPS: Key missing. Try again."
    SUCCESS_RESPONSE = {
        "text": {
            "additional_info": {
                "offloaded_model_info": {
                    "custom_model": {
                        "model_data": "null",
                        "model_path": "./custom_models/models/test_model/test_model.keras",
                        "num_classes": "test_model",
                        "num_layers": 5,
                        "save_path": "./custom_models/models/test_model/test_model.keras"
                    },
                    "layers_inference_time": [
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0
                    ],
                    "layers_sizes": [
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0
                    ],
                    "load_model_data": "true",
                    "model_analytics": {
                        "layer_0": {
                            "layer_inference_time": 10.0,
                            "layer_size": 10.0
                        },
                        "layer_1": {
                            "layer_inference_time": 10.0,
                            "layer_size": 10.0
                        },
                        "layer_2": {
                            "layer_inference_time": 10.0,
                            "layer_size": 10.0
                        },
                        "layer_3": {
                            "layer_inference_time": 10.0,
                            "layer_size": 10.0
                        },
                        "layer_4": {
                            "layer_inference_time": 10.0,
                            "layer_size": 10.0
                        }
                    },
                    "model_analytics_path": "./custom_models/models/test_model/analytics.json",
                    "model_name": "test_model",
                    "model_path": "./custom_models/models/test_model/test_model.keras",
                    "num_layers": 5,
                    "predictions": []
                },
                "offloading_algo_info": {
                    "avg_speed": 1472.0,
                    "best_offloading_layer": 3,
                    "inference_time_device": [
                        1.0,
                        1.8,
                        1.0,
                        1.8,
                        1.8,
                        9999999999.8
                    ],
                    "inference_time_edge": [
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0
                    ],
                    "layers_sizes": [
                        10.0,
                        10.0,
                        10.0,
                        10.0,
                        10.0
                    ],
                    "lowest_evaluation": 23.807,
                    "num_layers": 5
                }
            },
            "best_offloading_layer": 3
        }
    }


class OffloadingErrorSchema(Schema):
    message = fields.String()


class OffloadingModelInitializationSchema(Schema):
    model_name = fields.String(required=True)
    data = fields.String(required=True)
