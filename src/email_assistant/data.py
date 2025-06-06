import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import requests
from datasets import Dataset
from dvc_handler import DVCHandler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator

logger = logging.getLogger(__name__)


class EmailDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.tokenizer = self._setup_tokenizer()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dvc = DVCHandler(
            self.config["data"]["root_dir"],
            remote_path=self.config["data"]["remote_path"],
        )
        self._set_dvc()

        if config["data"]["pad_to_max_length"]:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8 if config["training"]["bf16"] else None,
            )

    def _setup_tokenizer(self) -> AutoTokenizer:
        """Set up and configure the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["model_name_or_path"],
            use_fast=self.config["model"]["use_fast_tokenizer"],
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _set_dvc(self) -> bool:
        """Set up DVC data management and ensure data availability."""

        train_path = Path(self.config["data"]["train_file"]).resolve()
        test_path = Path(self.config["data"]["test_file"]).resolve()

        in_data = train_path.exists() and test_path.exists()
        in_dvc = self.dvc.exists_in_dvc(train_path) and self.dvc.exists_in_dvc(
            test_path
        )

        if not in_data and not in_dvc:
            logging.info("No train/test found. Generating them")
            self._download_data()
            subprocess.run(["python", "summarize.py"])
            subprocess.run(["python", "split_data.py"])
            self.dvc.add_and_push(train_path)
            self.dvc.add_and_push(test_path)
        elif not in_data and in_dvc:
            logging.info("Local data missing. Pulling from DVC...")
            self.dvc.pull(train_path)
            self.dvc.pull(test_path)
        else:
            logging.info("Train/test already exist locally.")

    def _download_data(self) -> bool:
        """Download raw data from the configured URL."""

        url = self.config["data"]["download_url"]
        logger.info("Downloading data")

        try:
            response = requests.get(url)
            response.raise_for_status()
            raw_data_path = self.config["data"]["raw_data_path"]
            with open(raw_data_path, "wb") as f:
                f.write(response.content)

            print("\nDownload completed")

            self.dvc.add_and_push(Path(raw_data_path).resolve())
            return True

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return False

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages of training/testing."""

        if stage == "fit" or stage is None:
            self._setup_train_val()

        if stage == "test" or stage is None:
            self._setup_test()

    def _setup_train_val(self):
        """Set up training and validation datasets."""

        data = pd.read_csv(self.config["data"]["train_file"])

        train_data, val_data = train_test_split(
            data,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["seed"],
        )

        self.train_dataset = Dataset.from_pandas(train_data)
        self.val_dataset = Dataset.from_pandas(val_data)

        self.train_dataset = self.train_dataset.map(self._preprocess_function)
        self.val_dataset = self.val_dataset.map(self._preprocess_function)

    def _setup_test(self):
        """Setup test dataset"""
        test_data = pd.read_csv(self.config["data"]["test_file"])
        self.test_dataset = Dataset.from_pandas(test_data)

        self.test_dataset = self.test_dataset.map(
            lambda x: self._preprocess_function(x, is_test=True)
        )

    def _preprocess_function(self, sample, is_test=False):
        """Preprocess function for tokenizing samples using chat template

        Args:
            sample (Dict): Data sample
            is_test (bool, optional): Whether this is for test/inference. Defaults to False.

        Returns:
            Dict: Processed sample with tokenized inputs
        """

        max_seq_length = min(
            self.config["data"]["max_seq_length"], self.tokenizer.model_max_length
        )

        if is_test:
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that helps write emails.",
                },
                {"role": "user", "content": sample["summary"]},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            sample["text"] = text

            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

            sample["input_ids"] = tokenized["input_ids"].squeeze()
            sample["attention_mask"] = tokenized["attention_mask"].squeeze()
            return sample

        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that helps write emails.",
                },
                {"role": "user", "content": sample["summary"]},
                {"role": "assistant", "content": sample["email"]},
            ]

            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that helps write emails.",
                },
                {"role": "user", "content": sample["summary"]},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            full_tokenized = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

            prompt_tokenized = self.tokenizer(
                prompt_text, truncation=True, return_tensors="pt"
            )

            labels = full_tokenized["input_ids"].clone()
            prompt_length = len(prompt_tokenized["input_ids"][0])
            labels[0][:prompt_length] = -100

            sample["input_ids"] = full_tokenized["input_ids"].squeeze()
            sample["attention_mask"] = full_tokenized["attention_mask"].squeeze()
            sample["labels"] = labels.squeeze()

            return sample

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["per_device_train_batch_size"],
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.config["training"]["dataloader_num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["per_device_eval_batch_size"],
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.config["training"]["dataloader_num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["per_device_eval_batch_size"],
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.config["training"]["dataloader_num_workers"],
        )
