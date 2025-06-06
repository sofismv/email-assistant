from pathlib import Path

import hydra
import pandas as pd
import torch
from logging_utils import setup_logging
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class EmailSummarizer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = setup_logging(self.config)
        self.tokenizer = None
        self.model = None
        self.prompt_template = None

    def load_prompt_template(self):
        """Load prompt template from file"""
        prompt_path = Path(self.config.prompt_file)
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read().strip()
        self.logger.info(f"Loaded prompt template from {prompt_path}")

    def setup_model(self):
        """Initialize tokenizer and model with quantization"""
        self.logger.info(f"Loading tokenizer for model: {self.config.model.name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.quantization.bnb_4bit_compute_dtype,
        )

        self.logger.info(f"Loading model: {self.config.model.name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            use_cache=self.config.model.use_cache,
        )

        self.logger.info("Model and tokenizer loaded successfully")

    def load_emails(self):
        """Load emails from JSONL file"""
        input_path = Path(self.config.data.input_file)
        emails = pd.read_json(input_path, lines=True)
        self.logger.info(f"Loaded {len(emails)} emails from {input_path}")
        return emails

    def generate_summary(self, email_body: str) -> str:
        """Generate summary for a single email"""
        query = self.prompt_template.format(email_body=email_body)
        messages = [{"role": "user", "content": query}]

        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.model.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = outputs[0][input_ids.size(1) :]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        if output.lower().startswith("assistant\n\n"):
            output = output[len("assistant\n\n") :].strip()

        return output

    def process_emails(self):
        """Process all emails and generate summaries"""
        self.load_prompt_template()
        self.setup_model()
        emails = self.load_emails()

        summaries = []
        self.logger.info("Processing emails")

        for i in tqdm(range(len(emails)), desc="Generating summaries"):
            email_body = emails.iloc[i]["email"]
            summary = self.generate_summary(email_body)
            summaries.append(summary)

        emails["summary"] = summaries
        self.save_results(emails)
        self.logger.info("Email processing completed successfully")

    def save_results(self, emails_with_summaries: pd.DataFrame):
        """Save results to JSONL file"""
        output_path = Path(self.config.data.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        emails_with_summaries.to_json(
            output_path, orient="records", lines=True, force_ascii=False
        )

        self.logger.info(f"Results saved to {output_path}")


@hydra.main(
    version_base=None, config_path="../../config/summarization", config_name="default"
)
def main(cfg: DictConfig) -> None:
    """Main function to run email summarization"""
    summarizer = EmailSummarizer(cfg)
    summarizer.process_emails()


if __name__ == "__main__":
    main()
