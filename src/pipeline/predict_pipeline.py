import sys, os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_transformed=preprocessor.transform(features)
            preds=model.predict(data_transformed)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(
                self,
                footfall: int,
                TempMode: int,
                AirQuality: int,
                UltrSensData: int,
                CurSensRead: int,
                VolatileOrgComp: int,
                RPM: int,
                InptPressure: int,
                OptTemp: int
                ):

        self.footfall = footfall
        self.TempMode = TempMode
        self.AirQuality = AirQuality
        self.UltrSensData = UltrSensData
        self.CurSensRead = CurSensRead
        self.VolatileOrgComp = VolatileOrgComp
        self.RPM = RPM
        self.InptPressure = InptPressure        
        self.OptTemp = OptTemp

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "footfall": [self.footfall],
                "TempMode": [self.TempMode],
                "AirQuality": [self.AirQuality],
                "UltrSensData": [self.UltrSensData],
                "CurSensRead": [self.CurSensRead],
                "VolatileOrgComp": [self.VolatileOrgComp],
                "RPM": [self.RPM],
                "InptPressure": [self.InptPressure],
                "OptTemp": [self.OptTemp],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
