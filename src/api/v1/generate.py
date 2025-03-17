from http.client import HTTPException

from flask import Blueprint, request, jsonify

from src.config.config import MODEL_OUTPUT_DIR
from src.logging.logger import logger
from src.service.generator_service import GeneratorService

generate = Blueprint('generate', __name__)

@generate.route("/", methods=["POST"])
def generate_code():
    try:
        logger.info(f"Received request: {request.json}")

        data = request.json
        saved_model_path = data.get("saved_model_path", MODEL_OUTPUT_DIR)
        json_input = data.get("json_input", "")

        logger.info(f"Calling generator with params: [{saved_model_path}] [{json_input}]")
        generator_service = GeneratorService(saved_model_path=saved_model_path, json_input=json_input)
        generated_code = generator_service.generate_code()
        logger.info(f"Generation successful")

        return jsonify({"generated_code": generated_code})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(f"Generation failed with error {str(e)}")