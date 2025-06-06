import csv
import os
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_with_lora(cfg: DictConfig):
    """Load a quantized base model with LoRA adapters for inference.

    Args:
        cfg (DictConfig): Configuration

    Returns:
        tuple:
            - model: The loaded model with LoRA adapters in evaluation mode
            - tokenizer: The configured tokenizer
    """
    bnb_config = None
    if cfg.quantization.enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(
                torch,
                cfg.quantization.bnb_4bit_compute_dtype,
            ),
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        quantization_config=bnb_config,
        use_cache=cfg.use_cache,
    )

    model = PeftModel.from_pretrained(model, cfg.adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, cfg: DictConfig):
    """Takes a text prompt and generates an email response using
    the chat template format.

    Args:
        model: The loaded language model with LoRA adapters
        tokenizer: The tokenizer
        prompt (str): The input prompt
        cfg (DictConfig): Configuration

    Returns:
        str: The generated email content
    """
    messages = [
        {"role": "system", "content": "You are an assistant that helps write emails."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=cfg.generation.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][input_ids.size(1) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if output.lower().startswith("assistant\n\n"):
        output = output[len("assistant\n\n") :].strip()

    return output


def read_prompts_from_file(file_path: str):
    """Read prompts from file

    Args:
        file_path (str): Path to the input file containing prompts

    Returns:
        list: List of dictionaries containing:
            - id: Line number
            - prompt: The actual prompt text
    """
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                prompts.append({"id": line_num, "prompt": line})
    return prompts


def save_results(results, output_path: str):
    """Save generation results to a CSV file.

    Args:
        results (list): List of result dictionaries
        output_path (str): Path where the CSV file will be saved
    """
    """Save results in specified format"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Prompt", "Response"])
        for result in results:
            writer.writerow([result["id"], result["prompt"], result["response"]])


@hydra.main(version_base=None, config_path="../../config/infer", config_name="default")
def main(cfg: DictConfig):
    """Main function for email generation inference."""
    print("Loading model...")
    model, tokenizer = load_model_with_lora(cfg)
    print("Model loaded successfully!")

    if cfg.input_file:
        if not os.path.exists(cfg.input_file):
            raise FileNotFoundError(f"Input file not found: {cfg.input_file}")

        prompts = read_prompts_from_file(cfg.input_file)
        print(f"Found {len(prompts)} prompts to process")

        results = []
        for i, prompt_data in enumerate(prompts, 1):
            print(f"Processing prompt {i}/{len(prompts)}...")

            response = generate_response(model, tokenizer, prompt_data["prompt"], cfg)

            result = {
                "id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

            print(f"✓ Completed {i}/{len(prompts)}")

        output_file = getattr(
            cfg, "output_file", f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

        save_results(results, output_file)
        print("\nResults saved")

    else:
        prompt = input("Enter your prompt: ").strip()
        response = generate_response(model, tokenizer, prompt, cfg)

        print("\nGenerated Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    main()
