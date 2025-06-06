import logging
import os
from typing import Any, Dict

import evaluate
import pytorch_lightning as pl
import torch
from model import EmailAssistantModel
from transformers import get_scheduler

logger = logging.getLogger(__name__)


class EmailAssistantLightningModule(pl.LightningModule):
    """Module for training and evaluation models for the email generation task"""

    def __init__(self, config: Dict[str, Any], tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.config = config
        self.tokenizer = tokenizer
        self.model = EmailAssistantModel(config)
        self.metric = evaluate.load("bleu")
        self.test_responses = []

        self.lora_save_path = None

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.config["model"]["generation"],
            )

        generated_texts = []
        for input_ids_i, output_ids_i in zip(input_ids, outputs):
            new_tokens = output_ids_i[len(input_ids_i) :]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_texts.append(text)

        self.test_responses.extend(generated_texts)
        return {"generated_texts": generated_texts}

    def on_save_checkpoint(self, checkpoint):
        """Save only LoRA adapters, not the full model"""
        if hasattr(self.model.model, "save_pretrained"):
            checkpoint_dir = os.path.dirname(
                self.trainer.checkpoint_callback.best_model_path
            )
            lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
            os.makedirs(lora_dir, exist_ok=True)

            self.model.model.save_pretrained(lora_dir)
            self.lora_save_path = lora_dir

            checkpoint["lora_save_path"] = lora_dir

        return checkpoint

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        scheduler = get_scheduler(
            name=self.config["training"]["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=int(
                self.trainer.estimated_stepping_batches
                * self.config["training"]["warmup_ratio"]
            ),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}"
        )
