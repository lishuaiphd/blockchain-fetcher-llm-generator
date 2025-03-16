from transformers import AutoModelForCausalLM


class CausalLmLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_path)