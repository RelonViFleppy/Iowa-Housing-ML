import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, confusion_matrix

# --- 1. DATA INGESTION ---
# Download the dataset from Kaggle using the KaggleHub API
try:
    path = kagglehub.dataset_download('dansbecker/home-data-for-ml-course')
    iowa_file_path = os.path.join(path, "train.csv")
    
    # Check if the file exists, if not search in subdirectories
    if not os.path.exists(iowa_file_path):
        for root, dirs, files in os.walk(path):
            if "train.csv" in files:
                iowa_file_path = os.path.join(root, "train.csv")
                break

    iowa_data = pd.read_csv(iowa_file_path)
    print(f"Dataset loaded successfully from: {iowa_file_path}")
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

# --- 2. DATA PREPARATION ---
# Set the target for regression (LotArea)
y_lot = iowa_data['LotArea']

# Drop target and price to prevent data leakage, keeping only numeric features
X_lot = iowa_data.select_dtypes(include=[np.number]).copy()
cols_to_drop = [c for c in ['LotArea', 'SalePrice'] if c in X_lot.columns]
X_lot = X_lot.drop(cols_to_drop, axis=1)

# Split data into training and validation sets (80/20)
X_train_lot, X_val_lot, y_train_lot, y_val_lot = train_test_split(
    X_lot, y_lot, test_size=0.2, random_state=0
)

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_lot)
X_val_imputed = imputer.transform(X_val_lot)

# --- 3. REGRESSION ANALYSIS (Decision Tree & Random Forest) ---
# Default Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X_train_imputed, y_train_lot)
tree_mae = mean_absolute_error(y_val_lot, tree_model.predict(X_val_imputed))

# Random Forest Regressor for improved accuracy
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train_imputed, y_train_lot)
rf_mae = mean_absolute_error(y_val_lot, rf_model.predict(X_val_imputed))

print(f"\n--- Regression Results ---")
print(f"Decision Tree MAE: {tree_mae:.2f}")
print(f"Random Forest MAE: {rf_mae:.2f}")

# --- 4. CLASSIFICATION ANALYSIS ---
# Transform LotArea into a Binary target based on the median
lot_median = y_train_lot.median()
y_binary_train = (y_train_lot > lot_median).astype(int)
y_binary_val = (y_val_lot > lot_median).astype(int)

# Train a Decision Tree Classifier
clf_model = DecisionTreeClassifier(random_state=0)
clf_model.fit(X_train_imputed, y_binary_train)
clf_preds = clf_model.predict(X_val_imputed)

# Metrics calculation
acc = accuracy_score(y_binary_val, clf_preds)
prec = precision_score(y_binary_val, clf_preds)
rec = recall_score(y_binary_val, clf_preds)

print(f"\n--- Classification Report ---")
print(f"Accuracy:  {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")

# --- 5. VISUALIZATION ---
# Display the Confusion Matrix to evaluate classification performance

conf_mat = confusion_matrix(y_binary_val, clf_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - LotArea Classification')
plt.show()