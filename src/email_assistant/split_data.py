from pathlib import Path

import hydra
import pandas as pd
from logging_utils import setup_logging
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = setup_logging(self.config)

    def load_data(self) -> pd.DataFrame:
        """Load data from JSONL file"""
        input_path = Path(self.config.data.input_file)
        data = pd.read_json(input_path, lines=True)
        self.logger.info(f"Loaded {len(data)} records from {input_path}")
        return data

    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        train_data, test_data = train_test_split(
            data,
            test_size=self.config.split.test_size,
            random_state=self.config.split.random_state,
            shuffle=self.config.split.shuffle,
            stratify=None,
        )

        self.logger.info("Split data into:")
        self.logger.info(f"  - Train set: {len(train_data)})")
        self.logger.info(f"  - Test set: {len(test_data)})")

        return train_data, test_data

    def save_csv_files(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Save train and test data to CSV files"""
        output_dir = Path(self.config.data.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / self.config.data.train_file
        train_data.to_csv(train_path, index=False, encoding="utf-8")
        self.logger.info(f"Saved train data to {train_path}")

        test_path = output_dir / self.config.data.test_file
        test_data.to_csv(test_path, index=False, encoding="utf-8")
        self.logger.info(f"Saved test data to {test_path}")

    def process(self):
        """Main processing function"""
        data = self.load_data()
        train_data, test_data = self.split_data(data)
        self.save_csv_files(train_data, test_data)
        self.logger.info("Data splitting completed successfully!")


@hydra.main(version_base=None, config_path="../../config/data", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function to run data splitting"""
    splitter = DataSplitter(cfg)
    splitter.process()


if __name__ == "__main__":
    main()
