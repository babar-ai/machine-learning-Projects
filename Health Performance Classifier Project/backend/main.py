import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import Ensemble_models
import clean_data
import models

app = FastAPI()

@app.get("/")
def by():
    return {"message": "fast api is running"}

class HealthFeatures(BaseModel):
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    body_fat: float  # Will be mapped to 'body fat%'
    diastolic: int
    systolic: int
    grip_force: float  # Will be mapped to 'gripForce'
    sit_and_bend_forward_cm: float  # Will be mapped to 'sit and bend forward_cm'
    sit_ups_counts: int  # Will be mapped to 'sit-ups counts'
    broad_jump_cm: int  # Will be mapped to 'broad jump_cm'

@app.post("/predicate")
def predict_health(new_data: HealthFeatures):
    try:
    #     # Convert input data to DataFrame
        newdata_dict = new_data.dict()
        
        # # # Create mapping for column names
        column_mapping = {
            'body_fat': 'body fat_%',
            'grip_force': 'gripForce',
            'sit_and_bend_forward_cm': 'sit and bend forward_cm',
            'sit_ups_counts': 'sit-ups counts',
            'broad_jump_cm': 'broad jump_cm'
        }
        
        # Rename the keys in the dictionary before creating DataFrame
        for new_name, old_name in column_mapping.items():                       #items(): It allows you to access both keys and values simultaneously.
            if new_name in newdata_dict:
                newdata_dict[old_name] = newdata_dict.pop(new_name)
        
        newdata_frame = pd.DataFrame([newdata_dict])
        
        # Normalize the data
        newdata_scaled = clean_data.Normalize_newdata(newdata_frame)
        
        # OneHot Encoding for 'Gender' column
        newdata_encoded = clean_data.encoder_testdata(newdata_scaled)
        
        # Apply PCA transformation
        new_data_pca = models.Pca_test(newdata_encoded)
        
        # Get ensemble prediction
        ensemble_model = Ensemble_models.Ensemble_predication(newdata_encoded, new_data_pca)
        
        # Map numerical classes to labels
        class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        predicted_label = class_mapping[ensemble_model[0]]
        
        
        return {'prediction': f"The Health Class prediction is {predicted_label}"}
        
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Column mapping error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")