import argparse

from domain.generator import JavaGenerator
from model.optimizers.pruner import Pruner
from model.optimizers.quantizer import Quantizer
from model.trainers.gpt2trainer import Gpt2Trainer
from src.config.config import MODEL_OUTPUT_DIR, EPOCHS, BATCH_SIZE, DATASET_PATH
from src.model.loaders.gpt2loader import Gpt2Loader


def main(args):
    if args.train:
        fine_tuner = Gpt2Trainer(
            dataset_path=args.dataset_path,
            saved_model_path=args.saved_model_path
        )
        fine_tuner.train(output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size)
    elif args.generate:
        generator = JavaGenerator(saved_model_path=args.saved_model_path)
        generated_java_code = generator.generate_java(args.json_example)
        print(generated_java_code)
    elif args.prune:
        _model_loader = Gpt2Loader(args.saved_model_path)
        pruner = Pruner(_model_loader.load_model())
        if args.prune_method == "magnitude":
            pruner.magnitude_based()
        elif args.prune_method == "gradient":
            pruner.gradient_based()
        elif args.prune_method == "early_late":
            pruner.early_late_strategy()
        pruner.save_model(args.output_dir)
    elif args.quantize:
        _model_loader = Gpt2Loader(args.saved_model_path)
        quantizer = Quantizer(_model_loader.load_model())
        if args.quantization_method == "8bit":
            quantizer.apply_8bit()
        elif args.quantization_method == "4bit":
            quantizer.apply_4bit()
        quantizer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for JSON to Java code generation")

    # Training args
    parser.add_argument("--train", action="store_true", help="Set flag to train the model")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to the dataset file")
    parser.add_argument("--saved_model_path", type=str, help="Path to the saved fine-tuned model (optional)")
    parser.add_argument("--output_dir", type=str, default=MODEL_OUTPUT_DIR, help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")

    # Generation args
    parser.add_argument("--generate", action="store_true", help="Set flag to generate Java code")
    parser.add_argument("--json_example", type=str, help="JSON example for code generation")

    # Pruning args
    parser.add_argument("--prune", action="store_true", help="Set flag to apply pruning")
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "gradient", "early_late"], help="Pruning method to use")

    # Quantization args
    parser.add_argument("--quantize", action="store_true", help="Set flag to apply quantization")
    parser.add_argument("--quantization_method", type=str, choices=["8bit", "4bit"], help="Quantization method to use")

    args = parser.parse_args()
    main(args)
