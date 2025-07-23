import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        experience: int,
        age: int,
        
        education: str,
        location: str,
        job_title: str,
        gender: str
    ):
        self.experience = experience
        self.age = age
        
        self.education = education
        self.location = location
        self.job_title = job_title
        self.gender = gender


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Experience": [self.experience],
                "Age": [self.age],
                
                "Education": [self.education],
                "Location": [self.location],
                "Job_Title": [self.job_title],
                "Gender": [self.gender]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)