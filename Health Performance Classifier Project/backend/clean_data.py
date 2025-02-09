
import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

# Step 1: Download dataset
#heeloo
path = kagglehub.dataset_download("kukuroo3/body-performance-data")
print(f"Dataset downloaded to: {path}")

# Step 2: Verify the dataset file
dataset_file = os.path.join(path, 'bodyPerformance.csv')
print(f"Dataset file path: {dataset_file}")

# Step 3: Load the dataset
if os.path.exists(dataset_file):
    mydata = pd.read_csv(dataset_file)
    print("Dataset loaded successfully.")
else:
    print("Dataset file not found!")

# Step 4: Check for duplicates and missing values
dup = mydata.duplicated().sum()
miss_value = mydata.isnull().sum()
# print(f"Duplicate rows: {dup}")
# print(f"Missing values per column: {miss_value}")

# Drop duplicate rows
mydata.drop_duplicates(inplace=True)
mydata.reset_index(drop=True, inplace=True)

# Step 5: Handle outliers using z-scores
# numeric_columns = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic',
#                    'gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

#to detect numeric column automatically
numeric_columns = mydata.select_dtypes(include=['float64', 'int64']).columns

# Calculate z-scores
z_score_df = (mydata[numeric_columns] - mydata[numeric_columns].mean()) / mydata[numeric_columns].std()

# Function to remove outliers
def remove_outliers(z_scores_df, mydata, numeric_columns, z_thresh=3):
    all_outliers_indexes = set()                                            # Store all outlier indices

    # Identify outliers in each numeric column
    for column in z_scores_df.columns:
        outlier_indices = z_scores_df[z_scores_df[column].abs() > z_thresh].index.tolist()
        all_outliers_indexes.update(outlier_indices)

    # Drop rows with outlier indices
    mydata_cleaned = mydata.drop(index=list(all_outliers_indexes))
    
    # Reset the index after dropping rows
    mydata_cleaned.reset_index(drop=True, inplace=True)
    
    return mydata_cleaned

# Clean the data by removing outliers
clean_data = remove_outliers(z_score_df, mydata, numeric_columns)


# #lets normalize data

# def Normalize_data(new_data):
scaler = MinMaxScaler()
scaler.fit(clean_data[numeric_columns])
clean_data[numeric_columns] = scaler.transform(clean_data[numeric_columns])
    
def Normalize_newdata(newdata):
    
    non_numericdata = newdata.drop(columns = numeric_columns)
    newdata_normalized = pd.DataFrame(scaler.transform(newdata[numeric_columns]), columns=newdata[numeric_columns].columns)
    
    #to concatinate normalized part of columns and gender column
    newdata_normalized = pd.concat([newdata_normalized, non_numericdata.reset_index(drop=True)], axis=1)
    
    return newdata_normalized


#to split data 
X = clean_data.drop('class', axis=1)
y = clean_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# encoder = OneHotEncoder(drop='first', sparse_output=False)      # sparse_output=True it will save memory by storing only non-zero values.
encoder = joblib.load('gender_encoder.pkl')
def encoder_traindata(X_train):
    
    X_train['gender'] = X_train['gender'].replace({'Male': 'M', 'Female': 'F'})
    x_train_encoded = encoder.transform(X_train[['gender']])
    x_train_encoded_df = pd.DataFrame(x_train_encoded, columns= encoder.get_feature_names_out(['gender'])).astype('int')
    #to concatinate
    x_train_final = pd.concat([X_train.drop(['gender'], axis = 1).reset_index(drop=True), x_train_encoded_df.reset_index(drop=True)],axis=1)
    
    return x_train_final

def encoder_testdata(X_test):
    
    X_train['gender'] = X_train['gender'].replace({'Male': 'M', 'Female': 'F'})
    x_test_encoded = encoder.transform(X_test[['gender']])
    x_test_encoded_df = pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(['gender'])).astype('int')
    #to concatinate
    x_test_final = pd.concat([X_test.drop(['gender'], axis = 1).reset_index(drop=True), x_test_encoded_df.reset_index(drop=True)], axis=1)

    return x_test_final



#to encode the x and y 
X_train_encoded = encoder_traindata(X_train)
X_test_encoded = encoder_testdata(X_test)

#labal encoding 
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)





