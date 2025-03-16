from src.config.config import MODEL_OUTPUT_DIR
from src.domain.generator import JavaGenerator


class GeneratorService:
    def __init__(self, saved_model_path=MODEL_OUTPUT_DIR, json_input=""):
        self.saved_model_path = saved_model_path
        self.json_input = json_input

    def generate_code(self):
        generator = JavaGenerator(saved_model_path=self.saved_model_path)
        return generator.generate_java(self.json_input)