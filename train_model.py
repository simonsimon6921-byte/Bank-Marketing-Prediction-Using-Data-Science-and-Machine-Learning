from typing import Any

from numpy import dtype, ndarray
import pandas as pd
import pickle
from pandas.core.arrays.base import ExtensionArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
from scipy.io import arff
raw_data, meta = arff.loadarff("phpkIxskf.arff")
df = pd.DataFrame(raw_data)

# Encode categorical columns
le = LabelEncoder()
binary_cols = ["V5", "V14", "V8", "Class"]
for col in binary_cols :
    df[col] = le.fit_transform(df[col])
print(df[binary_cols].head())

# Features and target
x = df[["V14"]]
y = df["V5"]
# Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl - train_model.py:38")