import os
import sys
from src.Logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
import pandas as pd

from dataclasses import dataclass

## Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts', 'train.csv')
    test_data_path=os.path.join('artifacts', 'test.csv')
    raw_data_path=os.path.join('artifacts', 'raw.csv')

## Create the data ingestion CLass
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Start")

        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'train.csv'))
            logging.info('Dataset read as a pands Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data is created")

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info('Exception Occured in DataIngestion Stage')
            raise CustomException(e, sys)