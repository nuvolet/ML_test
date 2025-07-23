# Classification in Scikit-Learn

Classification is a supervised learning task where the goal is to predict discrete class labels for input data. Scikit-learn provides a comprehensive set of classification algorithms and tools.

## Common Classification Algorithms

### 1. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions
predictions = lr.predict(X_test)
probabilities = lr.predict_proba(X_test)  # Get probability estimates
```

### 2. Decision Trees
```python
from sklearn.tree import DecisionTreeClassifier

# Create and train decision tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)

# Feature importance
print("Feature importance:", dt.feature_importances_)
```

### 3. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Create and train random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

### 4. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

# Create and train SVM
svm = SVC(kernel='rbf', probability=True)  # Enable probability estimates
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
```

### 5. K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

# Create and train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

### 6. Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB

# Create and train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
```

## Complete Classification Example## Key Classification Concepts

### 1. **Model Evaluation Metrics**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Basic metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### 2. **Cross-Validation**

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 3. **Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 4. **Data Preprocessing**

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding categorical target variables
le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

## Types of Classification Problems

1. **Binary Classification**: Two classes (e.g., spam/not spam)
2. **Multiclass Classification**: Multiple classes (e.g., image classification)
3. **Multilabel Classification**: Multiple labels per instance

## Choosing the Right Algorithm

- **Logistic Regression**: Good baseline, interpretable, works well with linear relationships
- **Random Forest**: Handles non-linear relationships, robust to overfitting, provides feature importance
- **SVM**: Effective for high-dimensional data, good for non-linear problems with kernel trick
- **KNN**: Simple, good for local patterns, but can be slow on large datasets
- **Decision Trees**: Highly interpretable, but prone to overfitting
- **Naive Bayes**: Fast, works well with text data and small datasets

## Best Practices

1. **Always split your data** into training and testing sets
2. **Use cross-validation** to get robust performance estimates
3. **Scale your features** when using distance-based algorithms
4. **Handle imbalanced classes** with appropriate techniques
5. **Tune hyperparameters** using grid search or random search
6. **Evaluate using multiple metrics**, not just accuracy
7. **Visualize results** with confusion matrices and ROC curves

