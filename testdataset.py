import pandas as pd
import random
import numpy as np

# Create synthetic data
num_samples = 100  # Adjust the number of samples as needed
random.seed(42)  # Set a random seed for reproducibility

# Generate random values for features (Lifestyle, Diet, Age)
lifestyle_values = [random.choice(['Sedentary', 'Active', 'Highly Active']) for _ in range(num_samples)]
diet_values = [random.choice(['Vegetarian', 'Omnivorous']) for _ in range(num_samples)]
age_values = np.random.randint(18, 45, size=num_samples)

# Create a DataFrame
synthetic_data = pd.DataFrame({'Lifestyle': lifestyle_values, 'Diet': diet_values, 'Age': age_values})

# Generate synthetic values for Cycle_Length (you can modify this based on your model)
# Here, we're just using random values between 20 and 40
cycle_length_values = np.random.randint(20, 41, size=num_samples)

# Add Cycle_Length to the DataFrame
synthetic_data['Cycle_Length'] = cycle_length_values

# Save the synthetic data to a CSV file
synthetic_data.to_csv('synthetic_test_data.csv', index=False)
