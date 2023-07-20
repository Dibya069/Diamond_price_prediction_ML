import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.Logger import logging
from src.utils import save_obj, evaluate_mdoel
from dataclasses import dataclass

import sys, os

@dataclass
class ModelTrainerConfig:
    trained_model_trainer_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting Dependent and Independet variable form train and test data")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor()
            }   
            
            model_report: dict = evaluate_mdoel(x_train, y_train, x_test, y_test, models)
            print("\n=======================================================================")
            logging.info(f"Model Reports: {model_report}")

            ## To get the best model score forom the deictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"best model found, model name: {best_model_name}, R2_score: {best_model_score}")
            print("\n=========================================================================")
            logging.info(f"best model found, model name: {best_model_name}, R2_score: {best_model_score}")

            save_obj(

                file_path=self.model_trainer_config.trained_model_trainer_path,
                obj = best_model
            )


        except Exception as e:
            raise CustomException(e, sys)
