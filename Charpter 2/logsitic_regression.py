import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('./train.csv', encoding='utf-8')
data = data[data['native_country'] != ' ?']
none_scalar = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
label_encoder = LabelEncoder()
income = label_encoder.fit_transform(np.array(list(data['income'])))
data['income'] = income
onehot_encoded_dic = {}
for key in none_scalar:
    temp_data = np.array(list(data[key]))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(temp_data)
    onehot_encoded = to_categorical(integer_encoded)
    onehot_encoded_dic[key] = onehot_encoded

feature_list = data.columns.to_list()

data_set = []
for i in range(len(data)):
    temp_vector = []
    for j in range(len(feature_list)):
        if feature_list[j] in onehot_encoded_dic.keys():
            temp_vector.extend(onehot_encoded_dic[feature_list[j]][i])
        else:
            temp_vector.append(data.iloc[i, j])
    data_set.append(temp_vector)

data_set = np.array(data_set)
np.random.shuffle(data_set)

test_set = data_set[int(len(data_set) * 0.7):, :]
train_set = data_set[:int(len(data_set)*0.7), :]

trainX = train_set[:, :-1]
trainX = (trainX - np.mean(trainX, axis=0))/np.std(trainX, axis=0)
trainY = train_set[:, -1]

testX = test_set[:, :-1]
test_std = np.array([1e-8 if example == 0 else example for example in list(np.std(testX, axis=0))])
testX_normal = (testX - np.mean(testX, axis=0))/test_std
testY = test_set[:, -1]

def _shuffle(trainx, trainy):
    randomlist = np.arange(np.shape(trainx)[0])
    np.random.shuffle(randomlist)
    return trainx[randomlist], trainy[randomlist]

def sigmoid(z):
    res = 1/(1 + np.exp(-z))
    return np.clip(res, 1e-8, (1-(1e-8)))

def train(trainx, trainy):
    trainx = np.concatenate((np.ones((np.shape(trainx)[0], 1)), trainx), axis=1)
    epoch = 300
    batch_size = 32
    lr = 0.001
    w = np.zeros((np.shape(trainx)[1]))
    step_num = int(np.floor(len(trainx)/batch_size))
    cost_list = []
    for i in range(1, epoch):
        train_x, train_y = _shuffle(trainx, trainy)
        total_loss = 0.0
        for j in range(1, step_num):
            x = train_x[j*batch_size:(j+1)*batch_size, :]
            y = train_y[j*batch_size:(j+1)*batch_size]
            y_ = sigmoid(np.dot(x, w))
            loss = np.squeeze(y_) - y
            cross_entropy = -1 * (np.dot(y, np.log(y_)) + np.dot((1-y), np.log(1-y_)))/len(x)
            grad = np.dot(x.T, loss)/len(x)
            w = w - lr * grad

            total_loss += cross_entropy
        cost_list.append(total_loss)
        print("epoch:", i)
    return w, cost_list

def evaluation2(X, Y, w):
    X = np.concatenate((np.ones((np.shape(X)[0], 1)), X), axis=1)
    y = np.dot(X, w)
    y = np.array([1 if example > 0.5 else 0 for example in list(y)])
    error_count = 0
    for i in range(len(y)):
        if y[i] != Y[i]:
            error_count += 1

    error_rate = error_count/len(y)
    return error_rate


w, cost_list = train(trainX, trainY)
error_rate2 = evaluation2(trainX, trainY, w)
plt.plot(list(range(5, 10000)), cost_list[5:])