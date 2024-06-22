

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


