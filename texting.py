import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("loan_data.csv")

# Display initial rows
print(data.head())

# Check null values
print("\nNull Values Before Imputation:\n", data.isnull().sum())

# Fill missing values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    data[col].fillna(data[col].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mean(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)

print("\nNull Values After Imputation:\n", data.isnull().sum())

# Encode categorical features
label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Encode target
data['Loan_Status_encoded'] = label_encoder.fit_transform(data['Loan_Status'])

# ---------------------- EDA Plots ----------------------
# 1. Loan Status Count
sns.countplot(data=data, x='Loan_Status')
plt.title('Loan Approval Status Count')
plt.show()

# 2. Income Distribution by Education
sns.boxplot(data=data, x='Education', y='ApplicantIncome')
plt.title('Income Distribution by Education Level')
plt.show()

# 3. Loan Status by Property Area
sns.countplot(data=data, x='Property_Area', hue='Loan_Status')
plt.title('Loan Approval by Property Area')
plt.show()

# 4. Loan Amount by Dependents
sns.boxplot(data=data, x='Dependents', y='LoanAmount')
plt.title('Loan Amount Distribution by Number of Dependents')
plt.show()

# 5. Loan Amount Distribution
sns.histplot(data=data, x='LoanAmount', kde=True)
plt.title('Loan Amount Distribution')
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ---------------------- Model Training ----------------------
# Prepare features and target
X = data.drop(['Loan_Status', 'Loan_Status_encoded', 'Loan_ID'], axis=1)
y = data['Loan_Status_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model with class weight balanced
rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate
print("\n\u2705 Training Accuracy:", round(rf_model.score(X_train, y_train), 2))
print("\u2705 Test Accuracy:", round(rf_model.score(X_test, y_test), 2))

# Classification report
y_pred = rf_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# ROC-AUC
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
print("\nROC-AUC Score:", round(roc_auc_score(y_test, y_pred_prob), 2))

# Feature Importance
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title("Feature Importances")
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.show()

joblib.dump(rf_model, "loan_model.joblib")
print("✅ Model saved as loan_model.joblib")
