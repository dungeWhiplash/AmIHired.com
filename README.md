# AreYouHired


## Model's accuracy:
### Accuracy: 0.92
### Precision: 0.8588235294117647
### F1 Score: 0.8588235294117647
### Recall: 0.8588235294117647
### ROC AUC: 0.9286183310533515

## Algorithms used: 
### a) Logistic Regression
### b) Random Forest Tree

# As class imbalance is detected we will try implementing:
## 1. Oversampling 
  ### a) SMOTE
  ### b) ADASYN
  ### c) SMOTETomek
  ### d) SMOTEENN

## 2. Undersampling
  ### a) RandomUnderSampler

# We will also try implementing Principal Component Analysis for Dimensionality Reduction to see the impact

## Let's see the feature importance through Random Forest Classifier using SMOTETomek for class imbalances
![image](https://github.com/user-attachments/assets/5341972b-2d87-425a-84fe-d17581adb3c3)

# Steps to run the application:
## Install required python packages using:
### pip install -r requirements.txt
### Open your Anaconda Powershell Prompt:
### (base) > python hiring_str.py

## After it's successful running:
### (base) > streamlit run app.py

# Thank You and Happy Hiring!
