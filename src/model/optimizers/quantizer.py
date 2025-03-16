import torch

from transformers import BitsAndBytesConfig, AutoModelForCausalLM

class Quantizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def apply_8bit(self):
        if not torch.cuda.is_available():
            raise RuntimeError("8-bit quantization requires a GPU with CUDA support.")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype="auto"
        )

    def apply_4bit(self):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit quantization requires a GPU with CUDA support.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype="auto"
        )

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
