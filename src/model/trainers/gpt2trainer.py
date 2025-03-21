from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, AutoTokenizer

from config.config import MODEL_OUTPUT_DIR, LOGS_DIR, EPOCHS, BATCH_SIZE, MAX_LENGTH, DATASET_PATH
from data.loaders.dataloader import DataLoader
from model.loaders.causallmloader import CausalLmLoader


class Gpt2Trainer:
    def __init__(self, model_name="gpt2", dataset_path=DATASET_PATH, max_length=MAX_LENGTH, saved_model_path=None):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.saved_model_path = saved_model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lora_config = None
        self.configure_lora()

        if self.saved_model_path:
            _gpt2Loader = CausalLmLoader(self.saved_model_path)
            self.model = _gpt2Loader.load_model(hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        else:
            _gpt2Loader = CausalLmLoader(self.model_name)
            self.model = _gpt2Loader.load_model(hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            _data_loader = DataLoader(
                tokenizer=self.tokenizer,
                dataset_path=self.dataset_path,
                max_length=self.max_length
            )
            self.dataset = _data_loader.load_and_tokenize_dataset()

    def train(self, output_dir=MODEL_OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_dir=LOGS_DIR,
            save_strategy="epoch",
            eval_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset = self.dataset["validation"]
        )

        trainer.train()
        self.save_model(output_dir)

        trainer.evaluate(self.dataset["test"])

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