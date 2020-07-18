# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder  # depricated
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('dataset/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')

X = np.array(ct.fit_transform(X))

# Avoiding dummy variable trap
X = X[:, 1:]

# # # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# y_pred = regressor.predict(X_test)
#
# print(y_test - y_pred)
#
# # Backward elimnation
#
# # statsmodel library does not take into account b0 constant
# # need to add column of 1 in matrix of features
#
# X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
#
# # added column of 1s to matrix X, can also use statsmodels.tools.tools.add_constant(array)
# #
# # STEP 1: select significance level
# # SL = 0.05
#
# # STEP 2: fit the model with all possible predictors
# # Create optimal matrix of features
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # contains all independent variables
# X_opt = np.array(X_opt, dtype=float)
#
# # create new regressor from statsmodel
# # exog .. intercept is not included by default, needs to be added by the user (code at line 45)
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()  # ordinary least square
#
# c = list(regressor_OLS.pvalues)
# print(c, "<--- p values ")
# print(c.index(max(c)))
# # print(np.where(c == max(c)))
#
# regressor_OLS.summary()
#
# print(np.__version__)
# print(sm.__version__)
# print(pd.__version__)
#

# \begin{align}
# \dot{y} & = a_0x_0 +  a_1x_1 + a_2x_2 + a_3x_3  \\
# \end{align}

# print(regressor_OLS)


# Automatic Backward Elimination
def backwardelimination(x, sl):
    regressor_OLS = sm.OLS(y, x).fit()
    for i in range(0, len(x[0])):
        c = list(regressor_OLS.pvalues)
        print(c, "<--- p values ")
        j = c.index(max(c))
        maxp = max(regressor_OLS.pvalues).astype(float)
        if maxp > sl:
            x = np.delete(x, j, axis=1)
            regressor_OLS = sm.OLS(y, x).fit()
    print(regressor_OLS.summary())
    return x


X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]].astype(float)
SL = 0.05  # Significance Level
X_Modeled = backwardelimination(X_opt, SL)
print(X_Modeled)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
