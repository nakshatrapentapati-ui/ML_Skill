import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/adult.csv"
df = pd.read_csv(url)

# Basic EDA
print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe(include="all"))

# Remove missing values
df = df.dropna()

# Feature Engineering
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 40, 60, 100],
    labels=["Young", "Adult", "Middle_Aged", "Senior"]
)

df["capital_net"] = df["capital_gain"] - df["capital_loss"]

# Encode target
le = LabelEncoder()
df["income_encoded"] = le.fit_transform(df["income"])

print("\nAfter Feature Engineering:")
print(df.head())

# Income distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="income")
plt.title("Income Distribution")
plt.show()

# Age distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Income vs Education
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="education", hue="income")
plt.xticks(rotation=45)
plt.title("Education vs Income")
plt.show()

print("\nFeature Engineering completed successfully!")