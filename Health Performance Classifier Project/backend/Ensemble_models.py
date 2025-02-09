import models
import clean_data
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

knn_model = models.knn_model
rf_model = models.rf_model
gb_model = models.gb_model

#lets find test_accuracies of each model

X_train_encoded = clean_data.X_train_encoded
y_train = clean_data.y_train
X_test_encoded = clean_data.X_test_encoded
y_test = clean_data.y_test
X_test_pca = models.X_test_pca
        
    
def Ensemble_predication(new_data, new_data_pca):

    knn_prob = knn_model.predict_proba(new_data_pca)
    rf_prob = rf_model.predict_proba(new_data)    
    gb_prob = gb_model.predict_proba(new_data) 


    #based on the test accuracy let assign weight to model
    rf_weight = 0.5
    knn_weight = 0.2
    gb_weight = 0.3 

    #combine probabilities using weighted Probabilities
    ensemble_predict = (
        rf_weight*rf_prob + 
        knn_weight*knn_prob +
        gb_weight*gb_prob
    )
    
    ensemble_prob = np.argmax(ensemble_predict, axis=1)
    
    return ensemble_prob

ensemble_prob = Ensemble_predication(X_test_encoded, X_test_pca)
    
accuracy = accuracy_score( y_test, ensemble_prob)
print(f'Ensemble accuracy: {accuracy}')




