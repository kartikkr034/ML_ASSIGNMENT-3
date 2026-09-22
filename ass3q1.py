import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Create synthetic dataset
data = {
    "Age": [22, 25, np.nan, 30, 28, 35, np.nan, 40],
    "Salary": [25000, 32000, 28000, np.nan, 45000, 52000, 48000, 65000],
    "Department": [
        "IT", "HR", "IT", "Finance",
        np.nan, "IT", "HR", "Finance"
    ],
    "Years of Experience": [1, 2, 3, 5, np.nan, 8, 6, 12],
    "Purchased": [0, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

# Separate features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Define columns
numeric_features = [
    "Age",
    "Salary",
    "Years of Experience"
]

categorical_features = [
    "Department"
]

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n--- Processed Training Data ---")
print(X_train_processed.toarray()
      if hasattr(X_train_processed, "toarray")
      else X_train_processed)

print("\n--- Processed Testing Data ---")
print(X_test_processed.toarray()
      if hasattr(X_test_processed, "toarray")
      else X_test_processed)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])