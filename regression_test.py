import random

import numpy as np
from sklearn.neural_network import MLPRegressor

from utils import prepareData

from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_maxer = StandardScaler()
data = np.load('Data/Glucose_sim_data.npy', allow_pickle=True)
# data = min_maxer.fit_transform(data)

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

l_svr = LinearRegression()

data = min_maxer.fit_transform(data)
X = data[:-1, 2:]
Y = data[1:, 2:]
Z = data[:-1, :2]
xyz = np.concatenate((X, Z, Y), axis=-1)
random.shuffle(xyz)
train_x, train_y = xyz[:6000, :12], xyz[:6000, 12:]
test_x, test_y = xyz[6000:, :12], xyz[6000:, 12:]

l_svr.fit(train_x, train_y)
# print(l_svr.score(train_x, train_y))
# print(l_svr.score(test_x, test_y))
pre_y = l_svr.predict(test_x)

import torch
criterion = torch.nn.MSELoss()
print(criterion(torch.from_numpy(pre_y), torch.from_numpy(test_y)))


reg = MLPRegressor(hidden_layer_sizes=(2048,), max_iter=1000)
reg.fit(train_x, train_y)
pre_y = reg.predict(test_x)
print(criterion(torch.from_numpy(pre_y), torch.from_numpy(test_y)))