from transformers import GPT2LMHeadModel


class Gpt2Loader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        return GPT2LMHeadModel.from_pretrained(self.model_path)