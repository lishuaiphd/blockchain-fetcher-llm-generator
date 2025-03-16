from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Tokenizer, Trainer, TrainingArguments

from src.config.config import MODEL_OUTPUT_DIR, LOGS_DIR, EPOCHS, BATCH_SIZE, MAX_LENGTH, DATASET_PATH
from src.data.loaders.dataloader import DataLoader
from src.model.loaders.gpt2loader import Gpt2Loader


class Gpt2Trainer:
    def __init__(self, model_name="gpt2", dataset_path=DATASET_PATH, max_length=MAX_LENGTH, saved_model_path=None):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.saved_model_path = saved_model_path

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lora_config = None
        self.configure_lora()

        if self.saved_model_path:
            _gpt2Loader = Gpt2Loader(self.saved_model_path)
            self.model = _gpt2Loader.load_model()
        else:
            _gpt2Loader = Gpt2Loader(self.model_name)
            self.model = _gpt2Loader.load_model()
            _data_loader = DataLoader(self.dataset_path, self.max_length, self.tokenizer)
            self.dataset = _data_loader.load_and_tokenize_dataset()

    def train(self, output_dir=MODEL_OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=LOGS_DIR,
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"]
        )

        trainer.train()
        self.save_model(output_dir)

    def configure_lora(self, r=8, lora_alpha=16, lora_dropout=0.1):
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)