import pandas as pd
import seaborn as sns
import missingno as msno_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb


train_data = pd.read_csv('F:\自学2020\PythonML_Code\Charpter 3\\titanic\\train.csv')
test_data = pd.read_csv('F:\自学2020\PythonML_Code\Charpter 3\\titanic\\test.csv')
train_data.describe()
test_data.describe()

plt.figure(figsize=(10, 8))
msno_plot.bar(train_data)

train_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_data.drop(['Name', 'PassengerId'], axis=1, inplace=True)

train_data['Age'].fillna(30, inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Age'].fillna(30, inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

for i in range(8):
    plt.subplot(241+i)
    sns.countplot(x=train_data.iloc[:, i])

train_data['Sex'].replace('male', 0, inplace=True)
test_data['Sex'].replace('male', 0, inplace=True)
test_data['Sex'].replace('female', 1, inplace=True)


test_data['Embarked'] = [0 if example == 'S' else 1 if example=='Q' else 2 for example in train_data['Embarked'].values.tolist()]
test_data['Embarked'] = [0 if example == 'S' else 1 if example=='Q' else 2 for example in train_data['Embarked'].values.tolist()]