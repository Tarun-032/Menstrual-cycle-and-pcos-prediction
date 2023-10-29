import pandas as pd
import random

# Create an empty list to store the test data
test_data = []

# Generate 150 records
for _ in range(150):
    test_record = {
        'Age': random.randint(25, 40),  # Random age between 25 and 40
        'Lifestyle': random.choice([5, 10, 15]),  # Random numeric value for Lifestyle
        'Diet': random.choice([1, 2]),  # Random numeric value for Diet
        'Cycle_Length': random.randint(25, 35),  # Random cycle length between 25 and 35
        'Flow_Intensity': round(random.uniform(1, 5)),  # Rounded random value between 1 and 5 for Flow_Intensity
        'Issueupdated': random.randint(1, 4)  # Random value between 1 and 4 for Issueupdated
    }
    test_data.append(test_record)

# Create a DataFrame for the test dataset
test_df = pd.DataFrame(test_data)

# Save the test dataset to a CSV file
test_df.to_csv('testdataset.csv', index=False)
