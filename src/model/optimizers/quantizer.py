import bitsandbytes as bnb
import torch


class Quantizer:
    def __init__(self, model):
        self.model = model

    def apply_8bit(self):
        self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = bnb.nn.Linear8bitLt.convert_module(self.model)

    def apply_4bit(self):
        self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.half()
        self.model = bnb.nn.Linear4bit.convert_module(self.model)

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
