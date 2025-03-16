from src.config.config import MODEL_OUTPUT_DIR, DATASET_PATH, EPOCHS, BATCH_SIZE
from src.model.trainers.gpt2trainer import Gpt2Trainer


class TrainerService:
    def __init__(self, dataset_path=DATASET_PATH, output_dir=MODEL_OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        trainer = Gpt2Trainer(dataset_path=self.dataset_path)
        trainer.train(output_dir=self.output_dir, epochs=self.epochs, batch_size=self.batch_size)