import pandas as pd
import numpy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import power_transform
from sklearn.metrics import mean_squared_error, r2_score

# IMPORT DATA, ADD COLUMNS

data = pd.read_csv('fish_weights.csv')
data['l1_squared'] = data['Length1']**2
data['width_squared'] = data['Width']**2
data['volume'] = (1/3) * (data['l1_squared']+data['width_squared']) * data['Height']
temp = data['Species'].copy()
data = data.drop('Category', axis=1)
data = pd.get_dummies(data,'Species', dtype=int)
data = pd.concat([data, temp], axis=1)

# EDA

cols = ['Height', 'Width', 'Length1', 'Length2', 'Length3',
       'l1_squared', 'width_squared', 'volume', ]

data['Weight'].iloc[40] = 0.001 # ALL VALS > 0 TO USE BOXCOX
data['Weight'] = power_transform(data[['Weight']], 'box-cox')

for col in cols:
    data[col] = power_transform(data[[col]], method='box-cox')
    
for col in cols:
    sns.histplot(data[col])
    plt.xlabel(col)
    plt.show()

for col in cols:
    sns.scatterplot(x=data[col], y=data['Weight'])
    plt.xlabel(col)
    plt.ylabel('Wt. (gms)')
    plt.show()

for col in cols:
    sns.boxplot(data=data,x='Species',y=col)
    plt.xlabel('Species')
    plt.ylabel(col)
    plt.show()

# BUILD THE MODEL

x_train, x_test, y_train, y_test = train_test_split(data[['volume']], 
                                                    data['Weight'])
model = LinearRegression(fit_intercept=False)
model.fit(x_train, y_train)
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)

print('R2 Score on train data ', r2_score(y_train, train_preds))
print('R2 Score on test data ', r2_score(y_test, test_preds))