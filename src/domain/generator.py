from transformers import pipeline, GPT2Tokenizer

from src.model.loaders.gpt2loader import Gpt2Loader


class JavaGenerator:
    def __init__(self, model_name="gpt2", saved_model_path=None):
        self.model_name = model_name
        self.saved_model_path = saved_model_path

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        _gpt2Loader = Gpt2Loader(self.saved_model_path)
        self.model = _gpt2Loader.load_model()

    def generate_java(self, json_input):
        generator = pipeline("text-generation", model=self.saved_model_path, tokenizer=self.tokenizer)

        prompt = f"JSON: {json_input} \nJava:\n"
        output = generator(prompt, max_length=100)

        return output[0]["generated_text"]