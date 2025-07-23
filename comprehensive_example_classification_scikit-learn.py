import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
import warnings
warnings.filterwarnings('ignore')

# Load sample dataset (Wine dataset)
wine = load_wine()
X, y = wine.data, wine.target

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Classes: {wine.target_names}")
print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each classifier
results = {}

print("\n" + "="*60)
print("CLASSIFIER PERFORMANCE COMPARISON")
print("="*60)

for name, clf in classifiers.items():
    # Use scaled data for algorithms that benefit from it
    if name in ['Logistic Regression', 'SVM', 'KNN']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # Train the classifier
    clf.fit(X_train_use, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_use)
    y_pred_proba = clf.predict_proba(X_test_use) if hasattr(clf, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train_use, y_train, cv=5)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n{name}:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Find best performing classifier
best_clf_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest Classifier: {best_clf_name} (Accuracy: {results[best_clf_name]['accuracy']:.4f})")

# Detailed analysis of best classifier
print(f"\n" + "="*60)
print(f"DETAILED ANALYSIS: {best_clf_name}")
print("="*60)

best_pred = results[best_clf_name]['predictions']

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=wine.target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

# Feature importance (if available)
best_clf = classifiers[best_clf_name]
if hasattr(best_clf, 'feature_importances_'):
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': wine.feature_names,
        'importance': best_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# Hyperparameter tuning example for Random Forest
print(f"\n" + "="*60)
print("HYPERPARAMETER TUNING EXAMPLE (Random Forest)")
print("="*60)

rf_tuning = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search with cross-validation
grid_search = GridSearchCV(rf_tuning, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test the tuned model
best_rf = grid_search.best_estimator_
tuned_accuracy = best_rf.score(X_test, y_test)
print(f"Tuned model test accuracy: {tuned_accuracy:.4f}")

# Plotting results
plt.figure(figsize=(15, 10))

# Plot 1: Classifier comparison
plt.subplot(2, 3, 1)
names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in names]
colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

bars = plt.bar(range(len(names)), accuracies, color=colors)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Performance Comparison')
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom')

# Plot 2: Confusion Matrix
plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title(f'Confusion Matrix - {best_clf_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 3: Feature importance (if available)
if hasattr(best_clf, 'feature_importances_'):
    plt.subplot(2, 3, 3)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()

# Plot 4: Cross-validation scores
plt.subplot(2, 3, 4)
cv_means = [results[name]['cv_mean'] for name in names]
cv_stds = [results[name]['cv_std'] for name in names]

plt.errorbar(range(len(names)), cv_means, yerr=cv_stds, fmt='o', capsize=5)
plt.xlabel('Classifiers')
plt.ylabel('CV Accuracy')
plt.title('Cross-Validation Scores')
plt.xticks(range(len(names)), names, rotation=45, ha='right')

# Plot 5: Learning curves (for best classifier)
from sklearn.model_selection import learning_curve

plt.subplot(2, 3, 5)
train_sizes, train_scores, val_scores = learning_curve(
    best_clf, X_train_use, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title(f'Learning Curves - {best_clf_name}')
plt.legend()

plt.tight_layout()
plt.show()

# Model prediction example
print(f"\n" + "="*60)
print("MAKING PREDICTIONS ON NEW DATA")
print("="*60)

# Example: predict first 5 test samples
sample_data = X_test[:5]
sample_predictions = best_clf.predict(sample_data if best_clf_name not in ['Logistic Regression', 'SVM', 'KNN'] else X_test_scaled[:5])

if hasattr(best_clf, 'predict_proba'):
    sample_probabilities = best_clf.predict_proba(sample_data if best_clf_name not in ['Logistic Regression', 'SVM', 'KNN'] else X_test_scaled[:5])
    
    print("Sample Predictions with Probabilities:")
    for i in range(5):
        actual = wine.target_names[y_test[i]]
        predicted = wine.target_names[sample_predictions[i]]
        probs = sample_probabilities[i]
        
        print(f"Sample {i+1}:")
        print(f"  Actual: {actual}")
        print(f"  Predicted: {predicted}")
        print(f"  Probabilities: {dict(zip(wine.target_names, probs))}")
        print()

print("Classification analysis complete!")