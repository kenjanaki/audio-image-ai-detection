import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Paths to dataset
real_df = pd.read_csv(r"path\to\real_audio_features.csv")
fake_df = pd.read_csv(r"path\to\fake_audio_features.csv")

# Labeling real and fake audio
real_df['label'] = 0
fake_df['label'] = 1

# Combine real and fake audio data
combined_df = pd.concat([real_df, fake_df], ignore_index=True)

# Summary statistics for array columns
def summarize_array_features(df):
    array_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
                if df[col].apply(lambda x: isinstance(x, np.ndarray)).all():
                    array_cols.append(col)
                    df[col + '_mean'] = df[col].apply(np.mean)
                    df[col + '_std'] = df[col].apply(np.std)
                    df[col + '_min'] = df[col].apply(np.min)
                    df[col + '_max'] = df[col].apply(np.max)
                    df.drop(columns=[col], inplace=True)
            except Exception as e:
                print(f"Skipping column {col} due to error: {e}")
    print(f"Summarized columns: {array_cols}")
    return df

# Apply summary statistics
combined_df_summarized = summarize_array_features(combined_df)

# Drop unnecessary columns
combined_df_summarized.drop(columns=['file_name'], inplace=True, errors='ignore')

# Split into features and labels
X = combined_df_summarized.drop(columns=['label'])
y = combined_df_summarized['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set label distribution:\n{y_train.value_counts()}")
print(f"Test set label distribution:\n{y_test.value_counts()}")

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(eval_metric='logloss')

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Confusion matrix and ROC-AUC
print(confusion_matrix(y_test, y_pred))
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc}")

# Calculate Equal Error Rate (EER)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
frr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - frr)))]
eer = fpr[np.nanargmin(np.absolute((fpr - frr)))]
print(f"Equal Error Rate (EER): {eer}")
