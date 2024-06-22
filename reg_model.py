

# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.datasets import load_diabetes


# load data
diabetes = load_diabetes(as_frame=True)
diabetes.keys()

data = diabetes.data
features = diabetes.feature_names
targets = diabetes.target



#####################################################################################
##################### csv file

Df = diabetes['frame']
Df.info()
Df.isnull().sum()


##########################################################################################
################# data analysis

# quality + size
plt.rcParams['figure.figsize'] = [9,3]
plt.rcParams['figure.dpi'] = 300

# behaivor of data (classes, +Reg, -Reg)
sns.pairplot(Df,x_vars= Df.columns[:5],y_vars = 'target', height=4, aspect=1, kind='reg')  # first 5 elements
sns.pairplot(Df,x_vars= Df.columns[-5:],y_vars = 'target', height=4, aspect=1, kind='reg')  # second 5 elements


############################################################################################
############ Feature Scaling

# create x,y
x = Df.drop(['target'], axis = True)
y = Df['target']

from sklearn.preprocessing import StandardScaler

# standardisation
scaler = StandardScaler()
# apply scaler
x = scaler.fit_transform(x)





############################################################################################
############ split data set


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=0)

x_train.shape, y_train.shape
x_test.shape, y_test.shape


############################################################################################
############ Tree training 

# models import
from sklearn.tree import DecisionTreeRegressor

# classifier
Regression = DecisionTreeRegressor()

# fit
Regression.fit(x_train, y_train)

# predict
y_pred = Regression.predict(x_test)

# evaluation
from sklearn.metrics import mean_squared_error, r2_score

#RMSE
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
#r2_SCORE
print("r2_score ", r2_score(y_test, y_pred))

############################################################################################
############ visualize Tree

from sklearn import tree
plt.figure(figsize=(15,15), dpi=200)
ax = tree.plot_tree(Regression, feature_names=diabetes.feature_names, 
                    class_names=diabetes.target_filename,
                    filled=True)



############################################################################################
############ hyperparameter tuninig

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# hyperparameters set
params = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
          'splitter':['best', 'random'],
          'max_depth':[1,2,3]}


# Grid Search
grid = GridSearchCV(DecisionTreeRegressor(),
                    param_grid=params,
                    cv=5,
                    scoring='accuracy')

grid.fit(x_train, y_train)


# predict
y_pred = grid.predict(x_test)


# evaluation
from sklearn.metrics import mean_squared_error, r2_score

#RMSE
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
#r2_SCORE
print("r2_score ", r2_score(y_test, y_pred))



# Random Search
grid = RandomizedSearchCV(DecisionTreeRegressor(),
                    params,
                    cv=5,
                    scoring='accuracy')

grid.fit(x_train, y_train)




# predict
y_pred = grid.predict(x_test)


# evaluation
from sklearn.metrics import mean_squared_error, r2_score

#RMSE
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
#r2_SCORE
print("r2_score ", r2_score(y_test, y_pred))