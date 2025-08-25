import pandas as pd

# Load CSV
df = pd.read_csv("data/raw/fake_job_postings.csv")

# Print shape
print("Shape:", df.shape)

# Print first rows
print(df.head())

# Print target distribution
print(df['fraudulent'].value_counts())
