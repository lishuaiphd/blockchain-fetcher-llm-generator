from src.config.config import MODEL_OUTPUT_DIR, PRUNED_MODEL_OUTPUT_DIR
from src.model.loaders.gpt2loader import Gpt2Loader
from src.model.optimizers.pruner import Pruner


class PrunerService:
    def __init__(self, saved_model_path=MODEL_OUTPUT_DIR, output_dir=PRUNED_MODEL_OUTPUT_DIR, prune_method=""):
        self.saved_model_path = saved_model_path
        self.prune_method = prune_method
        self.output_dir = output_dir

    def prune_model(self):
        _data_loader = Gpt2Loader(self.saved_model_path)
        pruner = Pruner(_data_loader.load_model())

        if self.prune_method == "magnitude":
            pruner.magnitude_based()
        elif self.prune_method == "gradient":
            pruner.gradient_based()
        elif self.prune_method == "early_late":
            pruner.early_late_strategy()

        pruner.save_model(self.output_dir)