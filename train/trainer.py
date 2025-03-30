import functools
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb
from datasets import Dataset, load_dataset

from .self_dataset import load_and_process_dataset


class LLMTrainer:
    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        file_type: str = "json",
    ):
        self.model_name = model_name
        self.model = None
        self.dataset_path = dataset_path
        self.file_type = file_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.prefix = f"{model_name}_{dataset_path.split('.')[1].split('_')[-1]}"
        self.dataset = load_and_process_dataset(
            self.dataset_path, file_type=self.file_type
        )
        wandb.init(
            project="misinfo-training",
            name=self.prefix,
        )

    def preprocess_function(self, examples, tokenizer):
        return tokenizer(
            examples["input"], padding=True, truncation=True, max_length=512
        )

    @staticmethod
    def _compute_metric(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }
        
    def active_learning(
        self,
        unlabeled_dataset_path: str,
        file_type: str = "json",
        selection_size: int = 1000,
        retrain_epochs: int = 3,
        max_iterations: int = 10
    ):
        """
        Implements active learning by selecting the most uncertain samples from an unlabeled dataset
        and adding them to the training set iteratively.
        
        :param unlabeled_dataset_path: Path to the unlabeled dataset.
        :param file_type: File type of the dataset (default: "json").
        :param selection_size: Number of most uncertain samples to add to the training set per iteration.
        :param retrain_epochs: Number of epochs to retrain after adding new samples.
        :param max_iterations: Number of active learning iterations.
        """
        for iteration in range(max_iterations):
            print(f"Active learning iteration {iteration+1}/{max_iterations}")
            
            # Load and tokenize the unlabeled dataset
            unlabeled_dataset = load_dataset(file_type, data_files={"unlabeled": unlabeled_dataset_path})
            tokenized_unlabeled = unlabeled_dataset.map(
                functools.partial(self.preprocess_function, tokenizer=self.tokenizer),
                batched=True
            )

            # Get model logits for uncertainty estimation
            trainer = Trainer(model=self.model, tokenizer=self.tokenizer)
            predictions = trainer.predict(tokenized_unlabeled["unlabeled"]).predictions
            probabilities = F.softmax(torch.tensor(predictions), dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)

            # Select the most uncertain samples
            uncertain_indices = torch.argsort(entropy, descending=True)[:selection_size].tolist()
            uncertain_samples = [unlabeled_dataset["unlabeled"][i] for i in uncertain_indices]

            # Add new samples to the training dataset
            updated_train_data = self.dataset["train"].add_items(uncertain_samples)
            self.dataset["train"] = Dataset.from_dict(updated_train_data)

            # Retrain the model with the updated dataset
            self.train(num_train_epochs=retrain_epochs)

    def train(
        self,
        output_dir: str = "./train_res",
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        learning_rate: float = 5e-5,
        batch_size: int = 128,
        num_train_epochs: int = 10,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        metric_type: str = "f1",
        evaluation_steps: int = 300,
    ):
        
        tokenized_datasets = self.dataset.map(
            functools.partial(self.preprocess_function, tokenizer=self.tokenizer),
            batched=True,
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        num_labels = len(
            set(self.dataset["train"]["label"])
        )  # Determine number of classes dynamically
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{self.prefix}",
            learning_rate=learning_rate,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            eval_steps=evaluation_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=metric_type,
            logging_steps=logging_steps,
            gradient_accumulation_steps=2,
            bf16=True,
            report_to="wandb",  # Log to Weights & Biases
        )
        self.model = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metric,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ],  # Early stopping after 3 non-improving epochs
        )
        self.model.train()
        self.model.evaluate(eval_dataset=tokenized_datasets['test'])

    def evaluate(self, dataset_path: str, file_type: str = "json"):
        evalset = load_dataset(file_type, data_files={"test": dataset_path})
        evalset_tokenized = evalset.map(
            functools.partial(self.preprocess_function, tokenizer=self.tokenizer),
            batched=True
        )
        self.model.evaluate(eval_dataset=evalset_tokenized["test"])

    def finalize(self):
        wandb.finish()
