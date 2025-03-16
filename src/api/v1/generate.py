from http.client import HTTPException

from flask import Blueprint, request, jsonify

from config.config import MODEL_OUTPUT_DIR
from log.logger import logger
from service.generator_service import GeneratorService

generate = Blueprint('generate', __name__)

@generate.route("/", methods=["POST"])
def generate_code():
    try:
        logger.info(f"Received request: {request.json}")

        data = request.json
        saved_model_path = data.get("saved_model_path", MODEL_OUTPUT_DIR)
        prompt = data.get("prompt", "")

        logger.info(f"Calling generator with params: [{saved_model_path}] [{prompt}]")
        generated_code = GeneratorService(saved_model_path=saved_model_path, prompt=prompt).generate_code()
        logger.info(f"Generation successful")

        return jsonify({"generated_code": generated_code})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(f"Generation failed with error {str(e)}")