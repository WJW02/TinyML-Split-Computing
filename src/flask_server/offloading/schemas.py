from marshmallow import Schema, fields


class OffloadingSchema(Schema):
    model_name = fields.String(required=False)


class OffloadingErrorSchema(Schema):
    message = fields.String()
