from marshmallow import Schema, fields


class OffloadingSchema(Schema):
    device_id = fields.String(required=True, metadata= {'example': 'device-1', 'type': 'string'})
    model_name = fields.String(required=True)
    start_layer_index = fields.Integer(required=False)
    working_strategy = fields.String(required=False)
    algorithm_version = fields.String(required=False)


class OffloadingErrorSchema(Schema):
    message = fields.String()
