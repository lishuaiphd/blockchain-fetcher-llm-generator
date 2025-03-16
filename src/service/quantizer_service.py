from src.config.config import MODEL_OUTPUT_DIR, PRUNED_MODEL_OUTPUT_DIR
from src.model.loaders.gpt2loader import Gpt2Loader
from src.model.optimizers.quantizer import Quantizer


class QuantizerService:
    def __init__(self, saved_model_path=MODEL_OUTPUT_DIR, output_dir=PRUNED_MODEL_OUTPUT_DIR, quantization_method=""):
        self.saved_model_path = saved_model_path
        self.quantization_method = quantization_method
        self.output_dir = output_dir

    def quantize_model(self):
        _data_loader = Gpt2Loader(self.saved_model_path)
        quantizer = Quantizer(_data_loader.load_model())

        if self.quantization_method == "8bit":
            quantizer.apply_8bit()
        elif self.quantization_method == "4bit":
            quantizer.apply_4bit()

        quantizer.save_model(self.output_dir)