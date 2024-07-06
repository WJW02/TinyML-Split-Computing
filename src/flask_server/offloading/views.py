from flask import jsonify
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint
from flask_smorest import abort

from configs.configs import OffloadingApiConfigs, OffloadingApiMessages, OffloadingManagerConfigs
from flask_server.offloading.schemas import OffloadingSchema, OffloadingErrorSchema
from logger.Logger import Logger
from nn_model.model_manager import ModelManager
from offloading_tools.offloading_communication import OffloadingCommunicationHandler
from offloading_tools.offloading_manager import OffloadingManager

logger = Logger().get_logger(__name__)

offloading_blp = Blueprint(
    "offloading_blp",
    __name__,
    description=OffloadingApiConfigs.API_TITLE,
    url_prefix=OffloadingApiConfigs.OPENAPI_URL_PREFIX,
)

# Initialize communication handler
offloading_communication_handler = OffloadingCommunicationHandler()


@offloading_blp.route("/perform-offloading", methods=["POST"])
class OffloadingView(MethodView):
    @offloading_blp.response(status_code=200, description=OffloadingApiMessages.SUCCESS_RESPONSE,
                             schema=OffloadingSchema)
    @offloading_blp.response(status_code=500, description=OffloadingApiMessages.UNEXPECTED_ERROR,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.UNEXPECTED_ERROR})
    @offloading_blp.response(status_code=400, description=OffloadingApiMessages.NAME_KEY_MISSING,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.NAME_KEY_MISSING})
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

        if model_name is None or device_id is None:
            abort(400, description=OffloadingApiMessages.NAME_KEY_MISSING, message="Missing model_name or device_id")

        try:
            # Handles incoming message from device
            offloading_communication_handler.handle_incoming_message(message=body, device_id=device_id)

            # Initializes the offloading tool
            offloading_tool = OffloadingManager(
                algorithm_version=OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION,
                working_strategy=OffloadingManagerConfigs.DEFAULT_WORKING_STRATEGY,
                start_layer_index=OffloadingManagerConfigs.DEFAULT_START_LAYER_INDEX
            )

            # Initializes the model to be offloaded
            nn_model = ModelManager(model_name=model_name, model_path='./')

            # Gets the last message from the device
            device = offloading_communication_handler.device_manager.get_device(device_id)
            offloading_message = device.get_last_message()

            # Offloads the model
            result = offloading_tool.offload(offloading_message=offloading_message, model=nn_model)

            return jsonify({"text": result}), 200
        except Exception as e:
            abort(500, description=OffloadingApiMessages.UNEXPECTED_ERROR, message=str(e))


@offloading_blp.route("/get-offloading-communication-status", methods=["GET"])
class OffloadingCommunicationView(MethodView):
    @offloading_blp.response(status_code=200, description=OffloadingApiMessages.SUCCESS_RESPONSE,
                             schema=OffloadingSchema)
    @offloading_blp.response(status_code=500, description=OffloadingApiMessages.UNEXPECTED_ERROR,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.UNEXPECTED_ERROR})
    @offloading_blp.response(status_code=400, description=OffloadingApiMessages.NAME_KEY_MISSING,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.NAME_KEY_MISSING})
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
