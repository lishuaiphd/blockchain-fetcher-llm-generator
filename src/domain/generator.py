from transformers import pipeline, AutoTokenizer

from config.config import MAX_LENGTH, DO_SAMPLE, TEMPERATURE, TOP_K, TOP_P
from model.loaders.causallmloader import CausalLmLoader


class JavaGenerator:
    def __init__(self, model_name="gpt2", saved_model_path=None):
        self.model_name = model_name
        self.saved_model_path = saved_model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        _gpt2Loader = CausalLmLoader(self.saved_model_path)
        self.model = _gpt2Loader.load_model()

    def generate_java(self, prompt):
        generator = pipeline("text-generation", model=self.saved_model_path, tokenizer=self.tokenizer)
        formatted_prompt = f"Write Java code for: {prompt}\nJava Code:\n"

        output = generator(formatted_prompt,
                           max_length=MAX_LENGTH,
                           do_sample=DO_SAMPLE, # Enables sampling for more varied outputs
                           temperature=TEMPERATURE,  # Adjust for creativity (lower = more deterministic)
                           top_k=TOP_K,  # Consider top 50 words at each step
                           top_p=TOP_P)  # Nucleus sampling for diverse outputs
        generated_text = output[0]["generated_text"]

        return generated_text.replace(formatted_prompt, "").strip()
