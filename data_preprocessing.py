# importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 713842

df = pd.read_csv('dataset/Data.csv')

print(df)

X = df.values[:-1, :-3]

print(X)

y = df.values[:-1, 3]

print(y)

# from sklearn.preprocessing import Imputer

from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute = impute.fit(X[:, 1:3])  # upper bound is not considered so
X[:, 1:3] = impute.transform(X[:, 1:3])  # so give an extra allowance

print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)

# from sklearn.preprocessing import OneHotEncoder
#
# onehotencoder = OneHotEncoder(categories=X[0])
# X = onehotencoder.fit(X).toarray()
# print(X)

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('test train set ', X_train, X_test, y_train, y_test)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print('train', X_train)
print(X_test, 'test')
