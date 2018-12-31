import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_boston

boston = load_boston()

df = pd.DataFrame(data = boston.data, columns = boston.feature_names)
print(df.head())

#print(df.columns)
#print(boston.feature_names)
#print(boston.DESCR)
'''
sns.pairplot(df)
plt.show()
'''

X = df[['CRIM', 'ZN', 'INDUS', 'RM', 'B']]
y = df[['AGE']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

pred = lm.predict(X_test)

print(lm.coef_)
print(lm.intercept_)

plt.scatter(y_test, pred)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(mean_absolute_error(y_test, pred))
print(mean_squared_error(y_test, pred))
print(np.sqrt(mean_squared_error(y_test, pred)))

