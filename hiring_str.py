import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, r2_score, precision_score, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from collections import Counter
import ImbPipeline
import ColumnTransformer

df = pd.read_csv('recruitment_data.csv')
#print(df)

X = df.drop('HiringDecision',axis=1)
y = df['HiringDecision']

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

smote = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

rfc_model = RandomForestClassifier()

rfc_model.fit(X_train_resampled, y_train_resampled)

y_pred = rfc_model.predict(X_test_scaled)
y_pred_prob = rfc_model.predict_proba(X_test_scaled)[:,1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
print(f'ROC AUC: {roc_auc}')
probabilities = rfc_model.predict_proba(X_test)

feature_importances = rfc_model.feature_importances_
features = X_train.columns

plt.figure(figsize=(12, 8))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)  # Assuming all columns in X are numeric
    ]
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rfc_model)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipeline, 'model_pipeline.joblib')

# Predict and evaluate as before
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
print(f'ROC AUC: {roc_auc}')
