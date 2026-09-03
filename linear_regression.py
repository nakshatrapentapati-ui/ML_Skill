LINEAR REGRESSION
import pandas as pd
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("homeprices.csv")
sn.scatterplot(data=df, x='area', y='price')
reg = LinearRegression()
Train the model
reg.fit(df[['area']], df['price'])
Predict for 3300 sq.ft.
prediction_3300 = reg.predict(pd.DataFrame({'area': [3300]}))
print("Price for 3300 sq.ft:", prediction_3300[0])
Slope and intercept
print("Slope:", reg.coef_[0])
print("Intercept:", reg.intercept_)
Manual calculation
y = reg.coef_[0] * 3300 + reg.intercept_
print("Manual Prediction:", y)
Predict for 5000 sq.ft.
prediction_5000 = reg.predict(pd.DataFrame({'area': [5000]}))
print("Price for 5000 sq.ft:", prediction_5000[0])
R² Score
y_original = df['price']
y_predict = reg.predict(df[['area']])
R_square = r2_score(y_original, y_predict)
print("R² Score:", R_square)
sn.lmplot(data=df, x='area', y='price')
plt.show()