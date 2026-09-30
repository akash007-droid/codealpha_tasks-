# ==========================================
# CodeAlpha - Task 1
# Iris Flower Classification
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------
# 1. Load the dataset
# ------------------------------------------

df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------
# 2. Remove unnecessary column
# ------------------------------------------

# Iris.csv normally contains an Id column.
# It is not useful for classification.

if "Id" in df.columns:
    df = df.drop("Id", axis=1)


# ------------------------------------------
# 3. Separate features and target
# ------------------------------------------

X = df.drop("Species", axis=1)
y = df["Species"]


# ------------------------------------------
# 4. Split data into training and testing
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------
# 5. Train the Machine Learning model
# ------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ------------------------------------------
# 6. Make predictions
# ------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------
# 7. Evaluate the model
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("\nAccuracy:", accuracy)
print("\nAccuracy percentage:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------
# 8. Confusion Matrix
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Iris Flower Classification - Confusion Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------
# 9. Feature Importance
# ------------------------------------------

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(8, 5))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")

plt.tight_layout()
plt.show()


# ------------------------------------------
# 10. Example prediction
# ------------------------------------------

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\n================================")
print("EXAMPLE PREDICTION")
print("================================")

print("Input measurements:", sample)
print("Predicted species:", prediction[0])