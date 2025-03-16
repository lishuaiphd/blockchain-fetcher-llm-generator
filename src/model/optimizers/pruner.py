import torch
import torch.nn.utils.prune as prune


class Pruner:
    def __init__(self, model):
        self.model = model

    def magnitude_based(self, threshold=0.01):
        for layer in self.model.transformer.h:
            prune.l1_unstructured(layer.mlp.c_proj, name="weight", amount=threshold)

    def gradient_based(self, threshold=0.5):
        for layer in self.model.transformer.h:
            grad_norm = torch.norm(layer.mlp.c_proj.weight.grad).item()
            if grad_norm < 0.001:
                prune.l1_unstructured(layer.mlp.c_proj, name="weight", amount=threshold)

    def early_late_strategy(self, threshold=0.5):
        layers_to_prune = [0, 1, len(self.model.transformer.h) - 2, len(self.model.transformer.h) - 1]
        for idx in layers_to_prune:
            prune.l1_unstructured(self.model.transformer.h[idx].mlp.c_proj, name="weight", amount=threshold)

    def save_model(self, output_path):
        self.model.save_pretrained(output_path)

    def apply_pruning(self, strategy):
        if strategy == "magnitude":
            self.magnitude_based()
        elif strategy == "gradient":
            self.gradient_based()
        elif strategy == "early_late":
            self.early_late_strategy()
        else:
            raise ValueError("Unknown pruning strategy")
        print(f"Applied {strategy} pruning.")
