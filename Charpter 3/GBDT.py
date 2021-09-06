import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


wine_df = pd.read_csv('F:\自学2020\PythonML_Code\Charpter 3/winequality-red.csv', delimiter=';', encoding='utf-8')
columns_name = list(wine_df.columns)
for name in columns_name:
    q1, q2, q3 = wine_df[name].quantile([0.25, 0.5, 0.75])
    IQR = q3 - q1
    lower_cap = q1 - 1.5 * IQR
    upper_cap = q3 + 1.5 * IQR
    wine_df[name] = wine_df[name].apply(lambda x: upper_cap if x > upper_cap else (lower_cap if (x < lower_cap) else x))

wine_df = wine_df[wine_df['quality'] != 3.5]
wine_df = wine_df[wine_df['quality'] != 7.5]
wine_df['quality'] = wine_df['quality'].replace(8, 7)
wine_df['quality'] = wine_df['quality'].replace(3, 5)
wine_df['quality'] = wine_df['quality'].replace(4, 5)

sns.countplot(wine_df['quality'])
wine_df.describe()

trainX, testX, trainY, testY = train_test_split(wine_df.drop(['quality'], axis=1), wine_df['quality'], test_size=0.3, random_state=22)


model = GradientBoostingClassifier(n_estimators=430, learning_rate=0.1, subsample=0.8)
model.fit(trainX, trainY)
print("模型在训练集上分数为%s"%model.score(trainX, trainY))
pred_prob = model.predict_proba(trainX)
print('AUC:', metrics.roc_auc_score(np.array(trainY), pred_prob, multi_class='ovo'))
validX, tX, validY, tY = train_test_split(testX, testY, test_size=0.2)
print("模型在测试集上分数为%s"%model.score(validX, validY))
pred_prob = model.predict_proba(validX)
print('AUC test:', metrics.roc_auc_score(np.array(validY), pred_prob, multi_class='ovo'))


# GridSearchCV（网络搜索交叉验证）用于系统地遍历模型的多种参数组合，通过交叉验证从而确定最佳参数，适用于小数据集。
param_test1 = {'n_estimators': range(10, 501, 10), 'learning_rate': np.linspace(0.1, 1, 9)}
gsearch = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=1,
                                                            min_samples_split=2,
                                                            min_samples_leaf=1,
                                                            max_depth=3,
                                                            max_features=None,
                                                            subsample=0.8,
                                                            ), param_grid=param_test1, cv=5)
gsearch.fit(trainX, trainY)

means = gsearch.cv_results_['mean_test_score']
params = gsearch.cv_results_['params']
for i in range(len(means)):
    print(params[i], means[i])

print(gsearch.best_params_)
print(gsearch.best_score_)


param_test1 = {'max_depth': range(1, 6, 2), 'min_samples_split': range(1, 101, 10)}
gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=430,
                                                             learning_rate=1,
                                                             max_features=None,
                                                             min_samples_leaf=1,
                                                             subsample=1,
                                                             ), param_grid=param_test1, cv=5)
gsearch2.fit(trainX, trainY)
means = gsearch2.cv_results_['mean_test_score']
params = gsearch2.cv_results_['params']
for i in range(len(means)):
    print(params[i], means[i])

print(gsearch2.best_params_)
print(gsearch2.best_score_)


param_test1 = {'min_samples_leaf': range(1, 20, 2), 'min_samples_split': range(1, 101, 10)}
gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=430,
                                                             learning_rate=1,
                                                             max_features=None,
                                                             max_depth=9,
                                                             subsample=1,
                                                             ), param_grid=param_test1, cv=5)
gsearch3.fit(trainX, trainY)
