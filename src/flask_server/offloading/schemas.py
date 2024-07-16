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
    working_strategy = fields.String(required=False)


class OffloadingEvaluationMessages:
    WRONG_KEY = "OPS: Wrong API key. Try again."
    UNEXPECTED_ERROR = "ERROR: Unexpected error occurred!"
    MISSING_KEYS = "OPS: Key missing. Try again."
    SUCCESS_RESPONSE = "SUCCESS: Request processed successfully."


class OffloadingErrorSchema(Schema):
    message = fields.String()


class OffloadingModelInitializationSchema(Schema):
    model_name = fields.String(required=True)
    data = fields.String(required=True)
