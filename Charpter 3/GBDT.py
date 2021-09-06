import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


wine_df = pd.read_csv('E:\资料\PythonML_Code\Charpter 3/winequality-red.csv', delimiter=';', encoding='utf-8')
columns_name = list(wine_df.columns)
for name in columns_name:
    q1, q2, q3 = wine_df[name].quantile([0.25, 0.5, 0.75])
    IQR = q3 - q1
    lower_cap = q1 - 1.5 * IQR
    upper_cap = q3 + 1.5 * IQR
    wine_df[name] = wine_df[name].apply(lambda x: upper_cap if x > upper_cap else (lower_cap if (x < lower_cap) else x))

sns.countplot(wine_df['quality'])
wine_df.describe()

trainX, testX, trainY, testY = train_test_split(wine_df.drop(['quality'], axis=1), wine_df['quality'], test_size=0.3, random_state=22)


model = GradientBoostingRegressor(n_estimators=2000)
param_test1 = {'n_estimators': range(500, 1001, 50)}
gsearch = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_test1, cv=5)
gsearch.fit(trainX, trainY)
"""
这里有很多模型
"""
model.fit(trainX, trainY)
print("模型在训练集上的损失为%s"%model.score(trainX, trainY))
mean_squared_error(trainY, model.predict(trainX))


