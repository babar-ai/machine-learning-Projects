#to loaddata set from kaggle
import clean_data
import models
import pandas as pd
import Ensemble_models
import numpy as np


new_data = {
    'age': 27,
    'gender': 'M',
    'height_cm': 172.3,
    'weight_kg': 75.24,
    'body fat_%': 21.3,
    'diastolic': 80,
    'systolic': 130,
    'gripForce': 54.9,
    'sit and bend forward_cm': 18.4,
    'sit-ups counts': 60,
    'broad jump_cm': 217
}

#to convert dictionary to dataframe

newdata_frame = pd.DataFrame([new_data])  #wrapping the dictionary bcz The pd.DataFrame() function requires the input dictionary to be structured such that keys represent column names and values represent lists (or arrays) of data for those columns. 


newdata_scaled = clean_data.Normalize_newdata(newdata_frame)

#OneHot Encoding to encode 'Gender' column 
newdata_encoded = clean_data.encoder_testdata(newdata_scaled)

# print(f'columns of newdata_encoded {newdata_encoded.columns}')

#to apply pca
new_data_pca = models.Pca_test(newdata_encoded) #to find the predication of knn model only

#Let find the ensemble predication
ensemble_model = Ensemble_models.Ensemble_predication(newdata_encoded, new_data_pca)

#mapping numerical classes to labels
class_mapping = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D'}
predicted_label = class_mapping[ensemble_model[0]]

print(f'the predicated class is {predicted_label}')

