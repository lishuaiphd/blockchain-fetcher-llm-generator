from datasets import load_dataset, DatasetDict

from src.config.config import MAX_LENGTH, DATASET_PATH


def _split_dataset(full_dataset, train_size=0.8):
    train_validation_dataset = full_dataset['train']
    train_validation_split = train_validation_dataset.train_test_split(test_size=(1 - train_size))
    train_dataset = train_validation_split['train']
    validation_test_dataset = train_validation_split['test']
    test_val_split = validation_test_dataset.train_test_split(test_size=0.5)
    test_dataset = test_val_split['train']
    validation_dataset = test_val_split['test']

    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': validation_dataset,
    })

class DataLoader:
    def __init__(self, tokenizer, dataset_path=DATASET_PATH, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.max_length = max_length

    def load_and_tokenize_dataset(self):
        dataset = load_dataset(self.dataset_path)
        dataset = _split_dataset(dataset)
        return dataset.map(self._tokenize_function, batched=False)

    def _tokenize_function(self, example):
        java_code = "\n\n".join(example["java"])
        text = f"JSON: {example['json']} \nJava:\n{java_code}"
        tokenized_example = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)
        tokenized_example['labels'] = tokenized_example['input_ids']
        return tokenized_example
