#Titanic Survival — Full Logistic Regression Pipeline Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#load the dataset
df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)
print(df.columns)
#Target (y) = survived
#to understand the data set
print(df.info())
print(df.describe())
print(df.isnull().sum())#check missing values
#EDA
sns.countplot(x="survived", data=df)
plt.title("Titanic Survival Count")
plt.show()
#survival by gender
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival by Gender")
plt.show()
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()
#select the required features
features = ["pclass","sex","age","sibsp","parch","fare","embarked"]
X = df[features].copy()
y = df["survived"]
#for numer is age use median //missing values
X["age"] = X["age"].fillna(X["age"].median())
#for categorical colum use mode
X["embarked"] = X["embarked"].fillna(X["embarked"].mode()[0])
print(X.isnull().sum())
X = pd.get_dummies(X,columns=["sex", "embarked"],drop_first=True)
print(X.head())
#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    random_state=42,stratify=y)
scaler = StandardScaler()
#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
print("Model training completed")
y_pred = model.predict(X_test_scaled)
print(y_pred[:20])