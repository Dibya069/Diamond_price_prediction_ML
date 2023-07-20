from sklearn.impute import SimpleImputer ##Handle Missing value
from sklearn.preprocessing import StandardScaler    ## Handel Features scaling
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.Logger import logging
import sys, os
from dataclasses import dataclass
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfermation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")      

            ## Categorical col and Numerical Col
            num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_cols = ['cut', 'color', 'clarity']

            ## Define the coustomer raking for each categorical variable
            cut_category = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_category = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_category = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


            logging.info("Data Transformation Pipeline Initiated")
            ## Numeric Pipeline
            num_piplines = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            ## Categorical Pipeline
            cat_piplines = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_category, color_category, clarity_category])),
                    ('scaler', StandardScaler())
                ]
            )
            ##Processing Pipeline
            Preprocess = ColumnTransformer([
                ('num', num_piplines, num_cols),
                ('cat', cat_piplines, cat_cols)
            ])
            logging.info("Data Transformation Completed")

            return Preprocess


        except Exception as e:
            logging.info("Exception Occured In Data Transfermation(come to data_transformation.py file)")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read Train and Test Completed")
            logging.info(f"The dataframe Head:  \n{train_df.head().to_string()}")
            logging.info(f"The dataframe Head:  \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj = self.get_data_transformation_object()

            target_col = 'price'
            drop_col = [target_col, 'id']

            ## dividing dataset into dependent and independent
            ## Train data
            input_feature_train_df = train_df.drop(columns=drop_col, axis=1)
            traget_feature_train_df = train_df[target_col]

            ## Test data
            input_feature_test_df = test_df.drop(columns=drop_col, axis=1)
            traget_feature_test_df = test_df[target_col]

            ## Data Transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(traget_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(traget_feature_test_df)]

            save_obj(
                file_path = self.data_transfermation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transfermation_config.preprocessor_obj_file_path
            )

            logging.info("Applying Preprocessing object on training and test datasets.")


        except Exception as e:
            raise CustomException(e, sys)