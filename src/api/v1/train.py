from http.client import HTTPException

from flask import Blueprint, request, jsonify

from config.config import MODEL_OUTPUT_DIR, EPOCHS, BATCH_SIZE, DATASET_PATH
from log.logger import logger
from service.trainer_service import TrainerService

train = Blueprint('train', __name__)


@train.route("/", methods=["POST"])
def train_model():
    try:
        logger.info(f"Received request: {request.json}")

        data = request.json
        dataset_path = data.get("dataset_path", DATASET_PATH)
        output_dir = data.get("output_dir", MODEL_OUTPUT_DIR)
        epochs = data.get("epochs", EPOCHS)
        batch_size = data.get("batch_size", BATCH_SIZE)

        logger.info(f"Calling trainer with params: [{dataset_path}] [{output_dir}] [{epochs}] [{batch_size}]")
        TrainerService(dataset_path, output_dir, epochs, batch_size).train()
        logger.info("Training successful")

        return jsonify({"message": "Training completed", "output_dir": output_dir})
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(f"Training failed with error {str(e)}")