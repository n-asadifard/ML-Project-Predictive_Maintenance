from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                footfall=float(request.form.get('footfall')),
                TempMode=float(request.form.get('TempMode')),
                AirQuality=float(request.form.get('AirQuality')),
                UltrSensData=float(request.form.get('UltrSensData')),
                CurSensRead=float(request.form.get('CurSensRead')),
                VolatileOrgComp=float(request.form.get('VolatileOrgComp')),
                RPM=float(request.form.get('RPM')),
                InptPressure=float(request.form.get('InptPressure')),
                OptTemp=float(request.form.get('OptTemp'))
            )
            pred_df = data.get_data_as_data_frame()
            print("Sensor Input DataFrame:\n", pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Map prediction output (0: Normal/No Failure, 1: Failure Predicted)
            prediction_text = "Machine Failure Predicted!" if results[0] == 1 else "Machine Operating Normally."

            return render_template('home.html', results=prediction_text)
        
        except Exception as e:
            return render_template('home.html', results=f"Error occurred: {str.__(e) if 'e' in locals() else 'Invalid input'}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)