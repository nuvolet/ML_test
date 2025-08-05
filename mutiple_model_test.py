# 📦 Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 📥 Load dataset
url = "https://raw.githubusercontent.com/omairaasim/machine-learning-datasets/main/heart.csv"
df = pd.read_csv(url)

X = df.drop("target", axis=1)
y = df["target"]

# 📊 Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# ⚙️ Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 📈 Store results
results = {}
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Scale the features
        ('classifier', model)
    ])
    
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

# 📊 Plot the results
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(results))
plt.title("Model Comparison (Accuracy using 5-Fold CV)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
