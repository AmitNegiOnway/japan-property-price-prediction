import pandas as pd 
import numpy as np 
import os
import mlflow
from fastapi import File, UploadFile
from fastapi import FastAPI ,HTTPException
from schema import InputSchema,OutputSchema
from fastapi.middleware.gzip import GZipMiddleware
from predict import Predict


app = FastAPI()

@app.get('/Home')
def home():
    return {'message':'welcome to the Good Family'}

app.add_middleware(GZipMiddleware,minimum_size=1000)

@app.post('/predict_house_price',response_model=OutputSchema)
def Predict_Price(user_input:InputSchema):
    try:
        prediction=Predict(user_input)
        result=prediction.house_price_prediction()
        return {
            "Price": f"{int(result)} Yen"
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    





