from config.config import MODEL_OUTPUT_DIR
from domain.generator import JavaGenerator


class GeneratorService:
    def __init__(self, saved_model_path=MODEL_OUTPUT_DIR, prompt=""):
        self.saved_model_path = saved_model_path
        self.prompt = prompt

    def generate_code(self):
        generator = JavaGenerator(saved_model_path=self.saved_model_path)
        return generator.generate_java(self.prompt)