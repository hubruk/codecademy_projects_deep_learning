import pandas as pd

dataset = pd.read_csv('life_expectancy.csv')

dataset = dataset.drop(['Country'], axis = 1)
print(dataset.head())
print(dataset.describe())

labels = dataset.iloc[:,-1]
features= dataset.iloc[:,0:-1]

