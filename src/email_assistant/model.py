import logging
from typing import Any, Dict

import peft
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class EmailAssistantModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EmailAssistantModel
        Args:
            config (Dict[str, Any]): Configuration
        """
        super().__init__()
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        """Load and configure the transformer model with optional quantization and PEFT."""

        bnb_config = None
        if self.config["model"]["quantization"]["enabled"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config["model"]["quantization"]["load_in_4bit"],
                bnb_4bit_use_double_quant=self.config["model"]["quantization"][
                    "bnb_4bit_use_double_quant"
                ],
                bnb_4bit_quant_type=self.config["model"]["quantization"][
                    "bnb_4bit_quant_type"
                ],
                bnb_4bit_compute_dtype=getattr(
                    torch,
                    self.config["model"]["quantization"]["bnb_4bit_compute_dtype"],
                ),
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["model_name_or_path"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False,
        )

        if self.config["model"]["quantization"]["enabled"]:
            model = peft.prepare_model_for_kbit_training(model)

        model.gradient_checkpointing_enable()

        if self.config["model"]["peft"]["enabled"]:
            peft_config = peft.LoraConfig(
                r=self.config["model"]["peft"]["r"],
                lora_alpha=self.config["model"]["peft"]["lora_alpha"],
                lora_dropout=self.config["model"]["peft"]["lora_dropout"],
                target_modules=self.config["model"]["peft"]["target_modules"],
                task_type=self.config["model"]["peft"]["task_type"],
            )
            model = peft.get_peft_model(model, peft_config)

        return model

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
