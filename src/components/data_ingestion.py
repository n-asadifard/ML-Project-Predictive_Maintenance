import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'X_train.csv')
    test_data_path:str = os.path.join('artifacts', 'X_test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion has proceeded')
        try:
            sensor_data = pd.read_csv('notebook\dataset\data.csv')
            logging.info('Data has been read into sensor_data')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            sensor_data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('train test split has just been initiated')
            train_set, test_set = train_test_split(sensor_data, test_size=0.2, stratify=sensor_data["fail"], random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion and splitting the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()