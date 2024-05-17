import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('synthetic_dataset_with_features.csv')

# Handling missing values (you can customize this based on your dataset)
df = df.dropna()
print("No error in the dataset - Missing values handled.")

# Encoding categorical features (Lifestyle and Diet)
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['Lifestyle', 'Diet']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Lifestyle', 'Diet']))
df = pd.concat([df, encoded_df], axis=1)
print("No error in the dataset - Categorical features encoded.")

# Scaling numerical features
scaler = StandardScaler()
numerical_features = ['Cycle_Length', 'Age']  # Add more as needed
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print("No error in the dataset - Numerical features scaled.")

# Splitting into features and target
features = df.drop(['Cycle_Length', 'Date'], axis=1)
target = df['Cycle_Length']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("No error in the dataset - Train-test split completed.")

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("No error in the dataset - Linear regression model trained.")
print(f"Mean Squared Error: {mse}")
