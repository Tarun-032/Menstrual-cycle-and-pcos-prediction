import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the training data (replace with your actual training dataset)
training_data = pd.read_csv('/home/tarun003/Mini Project/synthetic_dataset_with_features.csv')

# Split data into features (X) and target variable (y)
X = training_data.drop(['Date','Cycle_Length'], axis=1)  # Exclude target and date columns
y = training_data['Cycle_Length']

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['Age']  # Add more as needed
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Encode categorical features
# encoder = OneHotEncoder(sparse=False)
# encoded_features = encoder.fit_transform(X[['Lifestyle', 'Diet']])
# encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Lifestyle', 'Diet']))
# X = pd.concat([X, encoded_df], axis=1)
print(X)

# Split the data into training and testing sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(model, 'linear_regression_model.pkl')
# joblib.dump(encoder, 'encoder.pkl')

length=model.predict([[3,4,15,2,3,35]])

print(length)
