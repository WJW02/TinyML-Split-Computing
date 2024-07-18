import base64

import numpy as np
from flask import jsonify
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint
from flask_smorest import abort

from configs.configs import OffloadingApiConfigs, OffloadingModelConfig
from flask_server.offloading.schemas import OffloadingEvaluationMessages, OffloadingErrorSchema, \
    OffloadingEvaluationSchema
from logger.Logger import Logger
from offloading_tools.offloading_communication import OffloadingCommunicationHandler
from offloading_tools.offloading_manager import OffloadingManager
from offloading_tools.offloading_model import OffloadingModel

logger = Logger().get_logger(__name__)

offloading_blp = Blueprint(
    "offloading_blp",
    __name__,
    description=OffloadingApiConfigs.API_TITLE,
    url_prefix=OffloadingApiConfigs.OPENAPI_URL_PREFIX,
)

# Initialize communication handler
offloading_communication_handler = OffloadingCommunicationHandler()


@offloading_blp.route("/evaluate", methods=["POST"])
class OffloadingEvaluationView(MethodView):
    @offloading_blp.response(status_code=200, description=OffloadingEvaluationMessages.SUCCESS_RESPONSE,
                             schema=OffloadingEvaluationSchema, example=OffloadingEvaluationMessages.SUCCESS_RESPONSE)
    @offloading_blp.response(status_code=500, description=OffloadingEvaluationMessages.UNEXPECTED_ERROR,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingEvaluationMessages.UNEXPECTED_ERROR})
    @offloading_blp.response(status_code=400, description=OffloadingEvaluationMessages.MISSING_KEYS,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingEvaluationMessages.MISSING_KEYS})
    def post(self):
        """
        A POST method to perform offloading of a Neural Network Model.

        Parameters:
            body (dict): A dictionary containing the model_name of the model to be offloaded.

        Returns:
            dict: A dictionary with a "text" key containing the response from the Offloading tool.

        Raises:
            400: If the provided model_name is missing in the request body.
            500: If an unexpected error occurs during the process.
        """
        body = request.get_json() or {}
        model_name = body.get("model_name")
        device_id = body.get("device_id")
        # Can be set but should always be 0 for offloading evaluation
        start_layer_index = body.get("start_layer_index")

        if not all([model_name, device_id, start_layer_index is not None]):
            abort(400, description=OffloadingEvaluationMessages.MISSING_KEYS,
                  message=OffloadingEvaluationMessages.MISSING_KEYS)

        try:
            # Handles incoming message from device
            offloading_communication_handler.handle_incoming_message(message=body, device_id=device_id)

            # Initializes the offloading tool
            offloading_tool = OffloadingManager(
                start_layer_index=start_layer_index
            )

            # Initializes the model to be offloaded
            offloading_model = OffloadingModel(
                model_name=model_name,
                model_path=OffloadingModelConfig.MODEL_PATH.replace("<MODEL_NAME>", model_name),
                model_analytics_path=OffloadingModelConfig.MODEL_ANALYTICS_PATH.replace("<MODEL_NAME>", model_name),
                load_model=True,
                load_model_data=True
            )

            # Gets the last message from the device
            device = offloading_communication_handler.device_manager.get_device(device_id)
            offloading_message = device.get_last_message()

            # Offloads the model
            result = offloading_tool.offload(
                offloading_message=offloading_message,
                model=offloading_model,
                device=device
            )
            return jsonify({"text": result}), 200
        except Exception as e:
            abort(500, description=OffloadingEvaluationMessages.UNEXPECTED_ERROR, message=str(e))


@offloading_blp.route("/communication-status", methods=["GET"])
class OffloadingCommunicationView(MethodView):
    def get(self):
        """
        A GET method to get the status of the offloading communication.

        Returns:
            dict: A dictionary with a "text" key containing the status of the offloading communication.

        Raises:
            500: If an unexpected error occurs during the process.
        """
        try:
            result = offloading_communication_handler.get_communication_status()
            return jsonify({"code": 200, "message": "Success", "communication_data": result}), 200
        except Exception as e:
            abort(500, description=e, message=str(e))


@offloading_blp.route("/model-initialization", methods=["POST"])
class OffloadingModelInitializationView(MethodView):
    def post(self):
        body = request.get_json() or {}
        model_name = body.get("model_name")
        model_data_b64 = body.get("model_data")
        shape = body.get("shape")

        # Decode the base64 string back to bytes and reconstruct the tensor
        model_data_bytes = base64.b64decode(model_data_b64)
        model_data_np = np.frombuffer(model_data_bytes, dtype=np.float32).reshape(shape)  # Adjust dtype as necessary

        offloading_model = OffloadingModel(
            model_name=model_name,
            model_path=OffloadingModelConfig.MODEL_PATH.replace("<MODEL_NAME>", model_name),
            model_analytics_path=OffloadingModelConfig.MODEL_ANALYTICS_PATH.replace("<MODEL_NAME>", model_name),
            load_model=True,
            load_model_data=False
        )

        try:
            result = offloading_model.perform_model_initialization(input_data=model_data_np)
            return jsonify({"text": str(result)}), 200
        except Exception as e:
            abort(500, description=OffloadingEvaluationMessages.UNEXPECTED_ERROR, message=str(e))


@offloading_blp.route("/model-inference", methods=["POST"])
class OffloadingInferenceView(MethodView):
    def post(self):
        body = request.get_json() or {}
        model_name = body.get("model_name")
        start_layer_index = body.get("start_layer_index")
        model_data_b64 = body.get("model_data")
        shape = body.get("shape")

        # Decode the base64 string back to bytes and reconstruct the tensor
        model_data_bytes = base64.b64decode(model_data_b64)
        model_data_np = np.frombuffer(model_data_bytes, dtype=np.float32).reshape(shape)  # Adjust dtype as necessary

        offloading_model = OffloadingModel(
            model_name=model_name,
            model_path=OffloadingModelConfig.MODEL_PATH.replace("<MODEL_NAME>", model_name),
            model_analytics_path=OffloadingModelConfig.MODEL_ANALYTICS_PATH.replace("<MODEL_NAME>", model_name),
            load_model=True,
            load_model_data=True
        )

        try:
            offloading_model.trigger_prediction(input_data=model_data_np, start_layer_index=start_layer_index,
                                                end_layer_index=None)
            result = offloading_model.predictions
            return jsonify({"text": str(result)}), 200
        except Exception as e:
            abort(500, description=OffloadingEvaluationMessages.UNEXPECTED_ERROR, message=str(e))
