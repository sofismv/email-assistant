import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from logging_utils import setup_logging
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from trainer import EmailAssistantLightningModule

from data import EmailDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmailAssistantTrainer:
    def __init__(self):
        self.logger = None
        self.config = None

    def train(self, cfg: DictConfig):
        """Main training function"""
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.logger = setup_logging(self.config)

        pl.seed_everything(self.config["seed"])

        data_module = EmailDataModule(self.config)
        model = EmailAssistantLightningModule(self.config, data_module.tokenizer)

        model.print_trainable_parameters()

        callbacks = self._setup_callbacks()

        mlflow_logger = None
        if self.config["mlflow_logger"]["enabled"]:
            mlflow_logger = MLFlowLogger(
                experiment_name=self.config["mlflow_logger"]["experiment_name"],
                run_name=self._get_run_name(),
                save_dir=self.config["mlflow_logger"]["output_log_dir"],
                tracking_uri=self.config["mlflow_logger"]["mlflow_tracking_uri"],
            )

        self.logger.info("Initializing trainer")
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["num_train_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            precision="16-mixed" if self.config["training"]["bf16"] else 32,
            accumulate_grad_batches=self.config["training"][
                "gradient_accumulation_steps"
            ],
            check_val_every_n_epoch=self.config["training"]["check_val_every_n_epoch"],
            callbacks=callbacks,
            logger=mlflow_logger,
            deterministic=False,
            enable_checkpointing=True,
        )

        if self.config["training"]["do_train"]:
            self.logger.info("Starting training")
            trainer.fit(model, data_module)
            self._save_final_lora_adapters(model)

        if self.config["training"]["do_predict"]:
            test_model = self._load_model_for_testing(trainer, data_module)

            self.logger.info("Starting testing")
            trainer.test(test_model, data_module)
            self._save_test_results(test_model.test_responses, data_module.test_dataset)

        self.logger.info("Training completed!")

    def _setup_callbacks(self):
        """Setup training callbacks"""

        callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.config["output_dir"], "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.2f}",
            save_top_k=self.config["training"]["save_total_limit"],
            monitor="val_loss",
            mode="min",
            save_weights_only=False,
            save_last=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

        early_stop_callback = EarlyStopping(
            monitor=self.config["early_stopping"]["monitor"],
            patience=self.config["early_stopping"]["patience"],
            mode=self.config["early_stopping"]["mode"],
            verbose=self.config["early_stopping"]["verbose"],
        )
        callbacks.append(early_stop_callback)

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        return callbacks

    def _save_final_lora_adapters(self, model):
        """Save final LoRA adapters after training completion"""
        final_lora_dir = os.path.join(self.config["output_dir"], "final_lora_adapters")
        os.makedirs(final_lora_dir, exist_ok=True)

        if hasattr(model.model.model, "save_pretrained"):
            model.model.model.save_pretrained(final_lora_dir)
            self.logger.info(f"Final LoRA adapters saved to {final_lora_dir}")

    def _get_run_name(self):
        """Generate run name"""
        return f"email_assistant_r{self.config['model']['peft']['r']}_lr{self.config['training']['learning_rate']}_epoch{self.config['training']['num_train_epochs']}"

    def _save_test_results(self, responses, test_dataset):
        """Save test results to CSV"""
        output_dir = os.path.join(self.config["output_dir"], "test_results")
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(
            {
                "summary": test_dataset["summary"],
                "model_response": responses,
            }
        )

        if "email" in test_dataset.column_names:
            df["email"] = test_dataset["email"]

        output_file = os.path.join(output_dir, f"{self._get_run_name()}.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Test results saved to {output_file}")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    trainer = EmailAssistantTrainer()
    trainer.train(cfg)


if __name__ == "__main__":
    main()
