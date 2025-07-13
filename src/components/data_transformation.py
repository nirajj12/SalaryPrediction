import sys
import os
from dataclasses import dataclass


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data transformation initiated")
            numeric_features = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

            categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
              ]
            )
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot_encoder', OneHotEncoder(sparse_output=False))
                
            ]    
            )

            logging.info("Numerical and categorical pipelines created")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numeric_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys) from e

    from sklearn.preprocessing import LabelEncoder

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Data Shape: {train_df.shape}")
            logging.info(f"Test Data Shape: {test_df.shape}")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "income"

            # Apply LabelEncoder to the target column
            label_encoder = LabelEncoder()
            train_df[target_column_name] = label_encoder.fit_transform(train_df[target_column_name])
            test_df[target_column_name] = label_encoder.transform(test_df[target_column_name])

            # Split input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Check for empty DataFrames
            if input_feature_train_df.empty:
                raise ValueError("Input feature DataFrame for training is empty.")
            if target_feature_train_df.empty:
                raise ValueError("Target feature DataFrame for training is empty.")

            logging.info("Applying preprocessing object on training and testing features")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert target Series to 2D arrays for concatenation
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Log shapes
            logging.info(f"Input Feature Train Array Shape: {input_feature_train_arr.shape}")
            logging.info(f"Target Feature Train Array Shape: {target_feature_train_arr.shape}")

            # Concatenate
            train_arr = np.concatenate([input_feature_train_arr, target_feature_train_arr], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_feature_test_arr], axis=1)

            logging.info("Saved preprocessing object")
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
            raise CustomException(e, sys)
