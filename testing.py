import pandas as pd
import joblib
from sklearn.discriminant_analysis import StandardScaler
from preprocessing import X_train

# Load the test data
test_data = pd.read_csv('testdataset.csv')

# Load the trained model and encoder
model = joblib.load('linear_regression_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = StandardScaler()

# Preprocess the test data as you did for the training data
# Standardize numerical features
numerical_features = ['Age']  # Add more as needed

# Fit the scaler to the training data
scaler.fit(X_train[numerical_features])

# Transform the test data using the fitted scaler
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Encode categorical features
encoded_features = encoder.transform(test_data[['Lifestyle', 'Diet']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Lifestyle', 'Diet']))
test_data = pd.concat([test_data, encoded_df], axis=1)

# Make predictions
predictions = model.predict(test_data)

# The 'predictions' variable now contains the predicted values for the target variable (Cycle_Length)      
print("Predicted Cycle Lengths:")
print(predictions)
