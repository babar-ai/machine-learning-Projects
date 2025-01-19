import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest


mydata = pd.read_csv(r"F://Machine Learning Hub//Machine Learning Specialization course//Unsupervised ML 3rdC//Projects//data.csv")
# print(mydata.head())

#Optional 
# Step 1: lets first convert the categorical data column into numerical using label encoding(Assigns a unique integer to each category.)
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'country']

label_encoding = {}

for col in categorical_cols:
    
    le = LabelEncoder()
    mydata[col] = le.fit_transform(mydata[col])

#to remove NULL values 
mydata.dropna(axis=1) 


# Step 2: Select only relevent features (numerical only) upon you want to apply anomaly detection algorithm
features = ['age',  'fnlwgt', 'capital-gain', 'hours-per-week', 'salary']
X = mydata[features]


# Step 3: Apply Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination= 'auto', random_state=42)   # if contamination=0.1 and you have 1000 data points, the algorithm will classify the 100 points with the highest anomaly scores as anomalies.
mydata['anomaly'] = iso_forest.fit_predict(X)

# Step 4: Interpret the results
# Anomalies are marked as -1, normal points as 1
print(mydata['anomaly'].value_counts())

#separate anomalies from normal data
anomalies = mydata[mydata['anomaly'] == -1]
normal_data = mydata[mydata['anomaly'] == 1]
    
print("Number of anomalies detected:", len(anomalies))
print("Sample anomalies:")
print(anomalies.head())

# Optional: Save anomalies to a CSV file
anomalies.to_csv(r"f:\Machine Learning Hub\anomalies.csv", index=False)

clean_data = mydata[mydata['anomaly'] == 1]


    
    
#Important 
'''
For each point, an anomaly score is computed based on the path length.
The score indicates how "anomalous" a point is:
A score close to 1 indicates a likely anomaly.
A score close to 0.5 suggests a normal point.

'''





