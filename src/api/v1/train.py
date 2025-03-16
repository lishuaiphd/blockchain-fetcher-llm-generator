from flask import Blueprint, request, jsonify

from src.config.config import MODEL_OUTPUT_DIR, EPOCHS, BATCH_SIZE, DATASET_PATH
from src.service.trainer_service import TrainerService

train = Blueprint('train', __name__)


@train.route("/", methods=["POST"])
def train_model():
    data = request.json
    dataset_path = data.get("dataset_path", DATASET_PATH)
    output_dir = data.get("output_dir", MODEL_OUTPUT_DIR)
    epochs = data.get("epochs", EPOCHS)
    batch_size = data.get("batch_size", BATCH_SIZE)

    trainer_service = TrainerService(dataset_path, output_dir, epochs, batch_size)
    trainer_service.train()

    return jsonify({"message": "Training completed", "output_dir": output_dir})