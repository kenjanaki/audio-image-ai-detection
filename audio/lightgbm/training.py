import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

# Load datasets
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()  # Flatten y_train to 1D array
y_test = pd.read_csv('y_test.csv').values.ravel()  # Flatten y_test to 1D array

# Remove non-numerical columns if present
X_train = X_train.drop(columns=['filename'], errors='ignore')
X_test = X_test.drop(columns=['filename'], errors='ignore')

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Define hyperparameters
params = {
    "objective": "binary",
    "boosting_type": "rf",  # Random forest-style boosting
    "num_leaves": 5,
    "force_row_wise": True,
    "learning_rate": 0.5,
    "metric": "binary_logloss",
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8
}

# Train the LightGBM model
num_round = 500
bst = lgb.train(params, train_data, num_boost_round=num_round, valid_sets=[test_data])

# Save the model to a file
bst.save_model('lgb_audio_model.txt')

# Make predictions
y_pred = bst.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1score = f1_score(y_test, y_pred_binary)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1score:.4f}")
