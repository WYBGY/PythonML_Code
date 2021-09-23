import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics



train_data = pd.read_csv('F:\自学2020\PythonML_Code\Charpter 3\\titanic\\train.csv')
test_data = pd.read_csv('F:\自学2020\PythonML_Code\Charpter 3\\titanic\\test.csv')

train_data.describe()
test_data.describe()

plt.figure(figsize=(10, 8))
msno_plot.bar(train_data)

train_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

train_data['Age'].fillna(30, inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Age'].fillna(30, inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

for i in range(8):
    plt.subplot(241+i)
    sns.countplot(x=train_data.iloc[:, i])

train_data['Sex'].replace('male', 0, inplace=True)
train_data['Sex'].replace('female', 1, inplace=True)
test_data['Sex'].replace('male', 0, inplace=True)
test_data['Sex'].replace('female', 1, inplace=True)


train_data['Embarked'] = [0 if example == 'S' else 1 if example=='Q' else 2 for example in train_data['Embarked'].values.tolist()]
test_data['Embarked'] = [0 if example == 'S' else 1 if example=='Q' else 2 for example in test_data['Embarked'].values.tolist()]


trainX, testX, trainy, testy = train_test_split(train_data.drop(['Survived'], axis=1), train_data['Survived'], test_size=0.2, random_state=10)

model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=6, min_child_weight=1, gamma=0, subsample=1,
                          objective='binary:logistic')
model.fit(trainX, trainy)
print(model.score(trainX, trainy))
model = xgb.XGBClassifier()

gsearch = GridSearchCV(estimator=model, param_grid={'n_estimators': range(10, 301, 10), 'max_depth': range(2, 7, 1)})
gsearch.fit(trainX, trainy)
means = gsearch.cv_results_['mean_test_score']
params = gsearch.cv_results_['params']
for i in range(len(means)):
    print(params[i], means[i])
print(gsearch.best_score_)
print(gsearch.best_params_)
# 0.8244262779474048
# {'max_depth': 5, 'n_estimators': 30}


# {'max_depth': 5}
# 0.8244262779474048


model2 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=30, max_depth=5, min_child_weight=1, gamma=0, subsample=1,
                           objective='binary:logistic', random_state=1)
gsearch = GridSearchCV(estimator=model2, param_grid={'gamma': np.linspace(0, 1, 11), 'subsample': np.linspace(0.1, 1, 10)})
gsearch.fit(trainX, trainy)
means = gsearch.cv_results_['mean_test_score']
params = gsearch.cv_results_['params']
for i in range(len(means)):
    print(params[i], means[i])
print(gsearch.best_score_)
print(gsearch.best_params_)

# 0.825824879346006
# {'gamma': 0.7, 'subsample': 0.7}

model3 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=30, max_depth=5, min_child_weight=1, gamma=0.7, subsample=0.7,
                           objective='binary:logistic', random_state=1, silent=False)
gsearch = GridSearchCV(estimator=model3, param_grid={'min_child_weight': range(1, 11)})
gsearch.fit(trainX, trainy)

# 0.825824879346006
# {'min_child_weight': 1}



model4 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=30, max_depth=5, min_child_weight=1, gamma=0.7, subsample=0.7,
                           objective='binary:logistic', random_state=1)
gsearch = GridSearchCV(estimator=model4, param_grid={'reg_lambda': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]})
gsearch.fit(trainX, trainy)

# 0.825824879346006
# {'reg_lambda': 1}

model4 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=30, max_depth=5, min_child_weight=1, gamma=0.7, subsample=0.7,
                           objective='binary:logistic', random_state=1)
gsearch = GridSearchCV(estimator=model4, param_grid={'learning_rate': [0.2, 0.5, 0.01, 0.001], 'n_estimators': [60, 150, 300, 3000]})
gsearch.fit(trainX, trainy)


model3 = xgb.XGBClassifier(learning_rate=0.01, n_estimators=300, max_depth=5, min_child_weight=1, gamma=0.7, subsample=0.7,
                           objective='binary:logistic', random_state=1, silent=False)

model3.fit(trainX, trainy, early_stopping_rounds=10, eval_metric='error', eval_set=[(testX, testy)])
print(model3.score(trainX, trainy))
print(model3.score(testX, testy))
metrics.accuracy_score(trainy, model3.predict(trainX))

# 0.8707865168539326
# 0.875


from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier(learning_rate=0.1, subsample=0.7)

gsearch = GridSearchCV(estimator=model3, param_grid={'n_estimators': range(10, 301, 10), 'learning_rate': np.linspace(0.1, 1, 10)})
gsearch.fit(trainX, trainy)
means = gsearch.cv_results_['mean_test_score']
params = gsearch.cv_results_['params']
for i in range(len(means)):
    print(params[i], means[i])
print(gsearch.best_score_)
print(gsearch.best_params_)

model3_1 = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, subsample=0.7)
gsearch = GridSearchCV(estimator=model3_1, param_grid={'max_depth': range(1, 9), 'min_samples_split': range(1, 21)})


model3_2 = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, subsample=0.7, max_depth=5, max_features=7, min_samples_leaf=31, min_samples_split=17)
gsearch = GridSearchCV(estimator=model3_2, param_grid={'max_features': range(3, 8, 1), 'subsample': np.linspace(0.1, 1, 10)})

model3_2.fit(trainX, trainy)
print(model3_2.score(trainX, trainy))
print(model3_2.score(testX, testy))

model3.fit(trainX, trainy)
model3.score(trainX, trainy)
model3.score(testX, testy)

