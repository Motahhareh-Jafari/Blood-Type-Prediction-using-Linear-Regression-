import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
data = pd.read_csv(r"location of healthcare_dataset.csv.zip")
df = pd.DataFrame(data)

df= pd.get_dummies(df, columns=['Blood Type'], drop_first=True)
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
print(data)

x = df[['Age' ,'Gender']]
y = df.filter(like='Blood Type')  

x_train, x_test, y_train, y_test= train_test_split(x, y , test_size=0.3 , random_state=30)

model = LinearRegression()
model.fit(x_train, y_train)
# print(y.columns)

print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)
y_pred = model.predict(x_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))


y_column = 'Blood Type_B-'  
plt.scatter(x['Gender'], y[y_column], color='blue')

line_x = np.linspace(x['Gender'].min(), x['Gender'].max(), 100)  
line_y = model.predict(pd.DataFrame({'Age': np.zeros(100), 'Gender': line_x}))[:, list(y.columns).index('Blood Type_B')]

plt.plot(line_x, line_y, color='red', label='Regression Line')
plt.xlabel('Gender')
plt.ylabel('Blood Type')
plt.title('Scatter Plot of Gender vs Blood Type')
plt.legend()
plt.show()

print(data.columns)