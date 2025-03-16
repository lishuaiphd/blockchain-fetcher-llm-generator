# Introduction
This is a very early work-in-progress prototype of a LLM. The goal is to generate Java code to perform proof-of-reserve
of a blockchain address.

Proof-of-reserve is used to verify ownership of a blockchain address. Otherwise said, the owner of the address has the
private key to the address. In the proof-of-reserve process, a challenge is sent to the owner of the address.
The owner then signs the challenge with the private key. The signature is verified with the public key.

The goal of the project is therefore to generate the Java code that will process the verification. The prompt is in
natural language. We assume the user does not have technical knowledge on the signature algorithm.

# Dataset
The prompt_to_signature_verifier_code_dataset.jsonl is only an example on how the dataset should be formatted.

# Examples of usage

## Launch server locally
python app.py --port 5000

## Build and launch container
docker build -t blockchain-verifier-generator .
docker run -p 5000:5000 blockchain-verifier-generator

## Training
python main.py --train --dataset_path "../data/prompt_to_signature_verifier_code_dataset.jsonl" --epochs 3 --batch_size 1 --output_dir "../model/gpt2-java-trained"

curl -X POST "http://127.0.0.1:5000/train" -H "Content-Type: application/json" \
     -d '{"dataset_path": "../data/prompt_to_signature_verifier_code_dataset.jsonl", "output_dir": "../model/gpt2-java-trained", "epochs": 3, "batch_size": 1}'

## Pruning
python main.py --prune --saved_model_path ../model/gpt2-java-trained --prune_method early_late --output_dir ../model/gpt2-java-pruned

curl -X POST "http://127.0.0.1:5000/prune" -H "Content-Type: application/json" \
     -d '{"saved_model_path": "../model/gpt2-java-trained", "prune_method": "early_late", "output_dir": "../model/gpt2-java-pruned"}'

## Quantization
python main.py --quantize --saved_model_path ../model/gpt2-java-trained --quantization_method 8bit --output_dir ../model/gpt2-java-quantized

curl -X POST "http://127.0.0.1:5000/quantize" -H "Content-Type: application/json" \
     -d '{"saved_model_path": "../model/gpt2-java-trained", "quantization_method": "8bit", "output_dir": "../model/gpt2-java-quantized"}'

## Generating
python main.py --generate --saved_model_path "../model/gpt2-java-trained" --prompt 'Generate a Solana verifier'

curl -X POST "http://127.0.0.1:5000/generate" -H "Content-Type: application/json" \
     -d '{"saved_model_path": "../model/gpt2-java-trained", "json_input": "'