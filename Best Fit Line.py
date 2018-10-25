import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Enter file name or url")

y = df['feature name']
X = df['dependent variable name']


X = X.values.reshape(len(X),1)
y = y.values.reshape(len(y),1)

# code for splitting the data into train and test. 
# Use test: 30% and train: 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#code to define the graph: title, axis notations and style
plt.scatter(X_test,y_test, color='black')
plt.title('Test Data')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit')

# code to fit the best line

regression = linear_model.LinearRegression()
regression.fit(X_train,y_train)
plt.plot(X_test, regression.predict(X_test), color ='blue')

plt.show()
