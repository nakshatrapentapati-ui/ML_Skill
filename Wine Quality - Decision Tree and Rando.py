# Wine Quality - Decision Tree and Random Forest Classification

# ==============================
# 1. Import Libraries
# ==============================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==============================
# 2. Load Dataset
# ==============================

df = pd.read_csv("winequality.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# ==============================
# 3. Check Missing Values
# ==============================

print("\nMissing Values:")
print(df.isnull().sum())


# ==============================
# 4. Create Classification Target
# ==============================

# Quality >= 7 → Good wine = 1
# Quality < 7  → Bad wine = 0

df["quality_class"] = df["quality"].apply(
    lambda x: 1 if x >= 7 else 0
)

print("\nClass Distribution:")
print(df["quality_class"].value_counts())


# ==============================
# 5. Separate Features and Target
# ==============================

X = df.drop(["quality", "quality_class"], axis=1)

y = df["quality_class"]


# ==============================
# 6. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==============================
# 7. Decision Tree Classifier
# ==============================

dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)


# ==============================
# 8. Decision Tree Evaluation
# ==============================

print("\n==============================")
print("DECISION TREE")
print("==============================")

print("Accuracy:")
print(accuracy_score(y_test, y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))


# ==============================
# 9. Random Forest Classifier
# ==============================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# ==============================
# 10. Random Forest Evaluation
# ==============================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print("Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# ==============================
# 11. Compare Both Models
# ==============================

dt_accuracy = accuracy_score(y_test, y_pred_dt)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

if rf_accuracy > dt_accuracy:
    print("\nRandom Forest performs better.")
elif dt_accuracy > rf_accuracy:
    print("\nDecision Tree performs better.")
else:
    print("\nBoth models have the same accuracy.")