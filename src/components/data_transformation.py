import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import SensorFeatureEngineering
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation        
        '''
        try:            
                                               
            preprocessor = Pipeline([("feature_engineering",SensorFeatureEngineering()),
                                     ("imputer", SimpleImputer(strategy='most_frequent')),
                                     ("scaler", StandardScaler())])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:            
            X_train=pd.read_csv(train_path)
            X_test=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            preprocessing_obj=self.get_data_transformer_object()
            logging.info("Preprocessing object instantiated")

            input_train = X_train.drop('Failure', axis=1)
            target_train = X_train["Failure"]
            input_test = X_test.drop('Failure', axis=1)
            target_test = X_test["Failure"]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_train)
            input_feature_test_arr = preprocessing_obj.transform(input_test)
            train_arr = np.c_[input_feature_train_arr, np.array(target_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test)]
            
            logging.info(f"Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)