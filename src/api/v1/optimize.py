from http.client import HTTPException

from flask import Blueprint, request, jsonify

from src.config.config import MODEL_OUTPUT_DIR, QUANTIZED_MODEL_OUTPUT_DIR, PRUNED_MODEL_OUTPUT_DIR
from src.logging.logger import logger
from src.service.pruner_service import PrunerService
from src.service.quantizer_service import QuantizerService

prune = Blueprint('prune', __name__)
quantize = Blueprint('quantize', __name__)

@prune.route("/", methods=["POST"])
def prune_model():
    try:
        logger.info(f"Received request: {request.json}")

        data = request.json
        saved_model_path = data.get("saved_model_path", MODEL_OUTPUT_DIR)
        output_dir = data.get("output_dir", PRUNED_MODEL_OUTPUT_DIR)
        prune_method = data.get("prune_method", "early_late")

        logger.info(f"Calling pruner with params: [{saved_model_path}] [{output_dir}] [{prune_method}]")
        pruner_service = PrunerService(saved_model_path, output_dir, prune_method)
        pruner_service.prune_model()
        logger.info("Pruning successful")

        return jsonify({"message": "Pruning completed", "output_dir": output_dir})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(f"Pruning failed with error {str(e)}")


@quantize.route("/", methods=["POST"])
def quantize_model():
    try:
        logger.info(f"Received request: {request.json}")

        data = request.json
        saved_model_path = data.get("saved_model_path", MODEL_OUTPUT_DIR)
        output_dir = data.get("output_dir", QUANTIZED_MODEL_OUTPUT_DIR)
        quantization_method = data.get("quantization_method", "8bit")

        logger.info(f"Calling quantizer with params: [{saved_model_path}] [{output_dir}] [{quantization_method}]")
        quantizer_service = QuantizerService(saved_model_path, output_dir, quantization_method)
        quantizer_service.quantize_model()
        logger.info("Quantization successful")

        return jsonify({"message": "Quantization completed", "output_dir": output_dir})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(f"Quantization failed with error {str(e)}")
