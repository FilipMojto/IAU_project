# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Define file paths for data
SESSION_FILE_PATH = "../resources/session.csv"

# Read session data into a DataFrame
df_sess = pd.read_csv(SESSION_FILE_PATH, delimiter='\t')

# Display information about the data
print(df_sess.info())

# Select features for training
selected_features = df_sess[['page_activity_duration', 'pct_rage_click', 'pct_scrandom']]

# Define X and y based on your available data
X = selected_features.copy()  # Create a copy to avoid "SettingWithCopyWarning"
y = df_sess['ack']            # 'ack' is target variable

numerical_features = ['page_activity_duration', 'pct_rage_click', 'pct_scrandom']

# Create an imputer for numerical features
numerical_imputer = SimpleImputer(strategy='mean')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])

# Standardize numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Create a logistic regression model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Display cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())
mean_accuracy = scores.mean()

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(scores)), scores, tick_label=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
plt.axhline(y=mean_accuracy, color='r', linestyle='--', label='Mean Accuracy')
plt.xlabel('Cross-Validation Folds')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.legend()
plt.show()