# Detailed roadmap for creating machine learning models in Python! Here's a comprehensive step-by-step guide:

# Phase 1: Foundation Setup

# 1. Environment Setup
# python
# Copy code
# Essential libraries to install
'''

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install tensorflow keras pytorch  # For deep learning
pip install plotly  # For interactive visualizations

'''


# 2. Project Structure
# css
# Copy code
# ml_project/
# ├── data/
# │   ├── raw/
# │   ├── processed/
# ├── notebooks/
# ├── src/
# │   ├── data_preprocessing.py
# │   ├── model.py
# │   ├── evaluation.py
# ├── models/
# ├── results/
# └── requirements.txt
# Phase 2: Data Pipeline

# 3. Data Collection & Loading
# python
# Copy code

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/raw/dataset.csv')
# or from database, API, etc.

# Initial data exploration
print(df.info())
print(df.describe())
print(df.head())

# 4. Exploratory Data Analysis (EDA)
# python
# Copy code
import matplotlib.pyplot as plt
import seaborn as sns

# Data distribution
df.hist(figsize=(15, 10))
plt.show()

# Correlation analysis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Missing values analysis
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# 5. Data Preprocessing
# python
# Copy code
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Handle missing values
df.fillna(df.mean(), inplace=True)  # or median, mode, forward fill

# Handle categorical variables
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Feature scaling
scaler = StandardScaler()
numerical_features = ['feature1', 'feature2', 'feature3']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Feature engineering
df['new_feature'] = df['feature1'] * df['feature2']

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Phase 3: Model Development

# 6. Model Selection & Training
# python
# Copy code

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Try multiple algorithms
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")
    
# 7. Hyperparameter Tuning
# python
# Copy code

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Example with Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# 8. Cross-Validation
# python
# Copy code

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Phase 4: Model Evaluation

# 9. Model Evaluation
# python
# Copy code

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Final predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # for binary classification

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 10. Feature Importance
# python
# Copy code
# Feature importance (for tree-based models)

feature_importance = best_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.show()

# Phase 5: Advanced Techniques

# 11. Deep Learning (Optional)
# python
# Copy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 12. Ensemble Methods
# python
# Copy code
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Phase 6: Deployment & Monitoring

# 13. Model Serialization
# python
# Copy code

import joblib
import pickle

# Save model
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Load model
loaded_model = joblib.load('models/best_model.pkl')
loaded_scaler = joblib.load('models/scaler.pkl')

# 14. Model Deployment
# python
# Copy code
# Create prediction function
def predict_new_data(new_data):
    # Preprocess new data
    new_data_scaled = loaded_scaler.transform(new_data)
    
    # Make prediction
    prediction = loaded_model.predict(new_data_scaled)
    probability = loaded_model.predict_proba(new_data_scaled)
    
    return prediction, probability

# Example usage
# new_sample = [[feature1_value, feature2_value, ...]]
# pred, prob = predict_new_data(new_sample)

# Phase 7: Best Practices

# 15. Code Organization
# python
# Copy code
# Create reusable classes
class MLPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def preprocess_data(self, X):
        # Preprocessing logic
        return processed_X
    
    def train(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
    
    def evaluate(self, X, y):
        # Evaluation logic
        pass
    
# 16. Monitoring & Logging
# python
# Copy code
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_with_logging():
    logger.info("Starting model training...")
    # Training code
    logger.info("Model training completed")

# Additional Tips:

# Version Control: Use Git for code versioning
# Data Versioning: Consider tools like DVC for data versioning
# Experiment Tracking: Use MLflow or Weights & Biases
# Documentation: Document your code and decisions
# Testing: Write unit tests for your functions
# Reproducibility: Set random seeds and save environment specifications