import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

iris = pd.read_csv("/Users/Olivia/Desktop/Week4-5project/iris.csv")

df = pd.DataFrame(iris)

df['variety'] = df['variety'].replace(['Setosa'], 0, regex=True)  # replace 'Setosa' with 0 for regression
df['variety'] = df['variety'].replace('Versicolor', 1, regex=True)  # replace 'Versicolor' with 1 for regression
df['variety'] = df['variety'].replace('Virginica', 2, regex=True)  # replace 'Virginica' with 2 for regression

print(df.shape)
print(df)

x = df.iloc[:, :4]
y = df['variety']

test = np.array([4.9, 3, 1.3, 0.2])
test = test.reshape(1, -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=100)
lm = LinearRegression()
lm.fit(x.values, y.values)

pickle.dump(lm, open('model.pickle', 'wb'))
model = pickle.load(open('model.pickle', 'rb'))

predictor = round(model.predict(test)[0])

print(predictor)
