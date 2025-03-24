from transformers import AutoModelForCausalLM, AutoConfig


class CausalLmLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self, hidden_dropout_prob=0, attention_probs_dropout_prob=0):
        config = AutoConfig.from_pretrained(self.model_path)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        return AutoModelForCausalLM.from_pretrained(self.model_path)