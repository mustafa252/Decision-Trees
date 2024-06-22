

# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.datasets import load_iris


# load data
iris = load_iris(as_frame=True)
iris.keys()

data = iris.data
features = iris.feature_names
targets = iris.target



#####################################################################################
##################### csv file

Df = iris['frame']
Df.info()
Df.isnull().sum()


##########################################################################################
################# data analysis

# quality + size
plt.rcParams['figure.figsize'] = [9,3]
plt.rcParams['figure.dpi'] = 300

# behaivor of data (classes, +Reg, -Reg)
sns.pairplot(Df, hue='target', height=3, kind='reg')


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

#startified
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

x_train.shape, y_train.shape
x_test.shape, y_test.shape


# check for stratify
y_train.value_counts()
y_test.value_counts() 



############################################################################################
############ Tree training 

# models import
from sklearn.tree import DecisionTreeClassifier

# classifier
classifier = DecisionTreeClassifier()

# fit
classifier.fit(x_train, y_train,)

# predict
y_pred = classifier.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot()



############################################################################################
############ visualize Tree

from sklearn import tree
plt.figure(figsize=(15,15), dpi=200)
ax = tree.plot_tree(classifier, feature_names=iris.feature_names, 
                    class_names=iris.target_names,
                    filled=True)



############################################################################################
############ hyperparameter tuninig

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# hyperparameters set
params = {'criterion': ['gini', 'entropy'],
          'splitter':['best', 'random'],
          'max_depth':[1,2,3]}


# Grid Search
grid = GridSearchCV(DecisionTreeClassifier(),
                    param_grid=params,
                    cv=5,
                    scoring='accuracy')

grid.fit(x_train, y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



# Random Search
grid = RandomizedSearchCV(DecisionTreeClassifier(),
                    params,
                    cv=5,
                    scoring='accuracy')

grid.fit(x_train, y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))