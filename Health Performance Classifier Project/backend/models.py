from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import clean_data
import joblib

X_train_encoded = clean_data.X_train_encoded
y_train = clean_data.y_train
X_test_encoded = clean_data.X_test_encoded
y_test = clean_data.y_test 

# print(X_train.columns)
# print(X_test.columns)

#to reduce the compotent fo the dataset to 8 as knn show it best accuracy at 8 features
pca = PCA(n_components=8)

def Pca_train(X_train):

    return pca.fit_transform(X_train)

def Pca_test(new_data):
    
    return pca.transform(new_data)

X_train_pca = Pca_train(X_train_encoded)
X_test_pca = Pca_test(X_test_encoded)

# #knn 
# knn_model = KNeighborsClassifier( weights='distance', p = 1, n_neighbors = 19)
# knn_model = knn_model.fit(X_train_pca, y_train)
# joblib.dump(knn_model, 'knn_model.pkl')

#to laod model
knn_model = joblib.load('knn_model.pkl')
#to predict
y_predict = knn_model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_predict)
print(f'accuracy of knn model {accuracy}')


#RandomForestClassifier model, lets load trained model

rf_model = joblib.load('rf_randomized.pkl')
print('rf model loaded')
# print(f'cloumns in X_test are {X_test.columns}')
# X_test['gender_M']
y_predict = rf_model.predict(X_test_encoded)
print('rf model predict')
accuracy = accuracy_score(y_test, y_predict)
print(f'accuracy of rf model {accuracy}')

#load gradient boosting classifier

gb_model = joblib.load('random_search_model.pkl')
y_predict = gb_model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_predict)
print(f'accuracy of gradientboosting model {accuracy}')
