from flask import jsonify
from flask.views import MethodView
from flask_smorest import Blueprint
from flask_smorest import abort

from flask_server.configs.configs import OffloadingApiConfigs, OffloadingApiMessages, OffloadingManagerConfigs
from flask_server.offloading.OffloadingManager import OffloadingManager
from flask_server.offloading.schemas import OffloadingSchema, OffloadingErrorSchema

offloading_blp = Blueprint(
    "offloading_blp",
    __name__,
    description=OffloadingApiConfigs.API_TITLE,
    url_prefix=OffloadingApiConfigs.OPENAPI_URL_PREFIX,
)


@offloading_blp.route("/perform-offloading", methods=["POST"])
class OffloadingView(MethodView):
    @offloading_blp.arguments(OffloadingSchema)
    @offloading_blp.response(status_code=200, description=OffloadingApiMessages.SUCCESS_RESPONSE,
                             schema=OffloadingSchema)
    @offloading_blp.response(status_code=500, description=OffloadingApiMessages.UNEXPECTED_ERROR,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.UNEXPECTED_ERROR})
    @offloading_blp.response(status_code=400, description=OffloadingApiMessages.NAME_KEY_MISSING,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.NAME_KEY_MISSING})
    @offloading_blp.response(status_code=422, description=OffloadingApiMessages.WRONG_NAME_KEY,
                             schema=OffloadingErrorSchema,
                             example={"message": OffloadingApiMessages.WRONG_NAME_KEY})
    def post(self, body):
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
        try:
            model_name = body.get("model_name")
            if model_name is None:
                abort(400, description=OffloadingApiMessages.NAME_KEY_MISSING)
        except KeyError as e:
            abort(400, description=OffloadingApiMessages.NAME_KEY_MISSING, details=str(e))

        try:
            offloading_tool = OffloadingManager(
                model_name=model_name,
                algorithm_version=OffloadingManagerConfigs.DEFAULT_ALGORITHM_VERSION,
                working_strategy=OffloadingManagerConfigs.DEFAULT_WORKING_STRATEGY
            )
            result = offloading_tool.offload()
            return jsonify({"text": result}), 200
        except Exception as e:
            abort(500, description=OffloadingApiMessages.UNEXPECTED_ERROR, details=str(e))