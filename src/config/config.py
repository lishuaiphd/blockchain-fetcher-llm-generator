# Paths
LOGS_DIR = "../logs"
DATASET_PATH = "../data/prompt_to_signature_verifier_code_dataset.jsonl"
MODEL_OUTPUT_DIR = "../model/gpt2-java-trained"
PRUNED_MODEL_OUTPUT_DIR = "../model/gpt2-java-pruned"
QUANTIZED_MODEL_OUTPUT_DIR = "../model/gpt2-java-quantized"

# Training
EPOCHS = 3
BATCH_SIZE = 1
MAX_LENGTH = 512

# Prediction
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9