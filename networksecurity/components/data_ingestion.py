
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_collection_as_dataframe(self):
        """Load the course dataset from the local CSV file."""
        try:
            csv_path = os.path.join(
                os.getcwd(), "Network_Data", "phisingData.csv"
            )

            if not os.path.isfile(csv_path):
                raise FileNotFoundError(
                    f"Dataset not found: {csv_path}. "
                    "Run main.py from the project root."
                )

            df = pd.read_csv(csv_path)

            if df.empty:
                raise ValueError("The dataset contains no rows.")

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Loaded local dataset: {df.shape}")
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dataframe.to_csv(path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            logging.info(
                f"Train rows: {len(train_set)}, test rows: {len(test_set)}"
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            return DataIngestionArtifact(
                trained_file_path=(
                    self.data_ingestion_config.training_file_path
                ),
                test_file_path=(
                    self.data_ingestion_config.testing_file_path
                ),
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
