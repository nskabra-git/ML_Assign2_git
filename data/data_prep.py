"""
# The UCI dataset @ https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# have two seperate file and not in csv format
# # 1. wdbc.data - contains the actual data
# # 2. wdbc.names - contains the feature title.
# We need to merge to form a clean dataset.
# This file performs the task.
"""

import pandas as pd

dataset_name = "breast_cancer"
dataset_source = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"

columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv("breast+cancer+wisconsin+diagnostic/wdbc.data", header=None, names=columns)
df.to_csv("wdbc.csv", index=False)

# Print All Metadata Fields
print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Dataset: {dataset_name}")
print(f"Source: {dataset_source}")
#print(f"Total Samples: {n_samples}")
#print(f"Number of Features: {n_features}")
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
# Display first few rows
df.head()
