import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import torch

standarder = StandardScaler()
min_maxer = MinMaxScaler()

# 标准化
data1 = np.load('Data/Glucose_sim_data002.npy', allow_pickle=True)
data1 = standarder.fit_transform(data1)
x1 = data1[:-1, :]
y1 = data1[1:, 2:]
xy1 = np.concatenate((x1, y1), axis=-1)
random.shuffle(xy1)
x1 = xy1[:, :-12]
y1 = xy1[:, -12:]
y1_tensor = torch.tensor(y1, dtype=torch.float32)

# min-max 归一化
data2 = np.load('Data/Glucose_sim_data002.npy', allow_pickle=True)
data2 = standarder.fit_transform(data2)
x2 = data2[:-1, :]
y2 = data2[1:, 2:]
xy2 = np.concatenate((x2, y2), axis=-1)
random.shuffle(xy2)
x2 = xy2[:, :-12]
y2 = xy2[:, -12:]
y2_tensor = torch.tensor(y2, dtype=torch.float32)

# data2 = np.load('Data/Glucose_sim_data002.npy', allow_pickle=True)
# data2 = min_maxer.fit_transform(data2)
# x2 = data2[:-1, :]
# y2 = data2[1:, 2:]
# y2_tensor = torch.tensor(y2, dtype=torch.float32)
#
# data3 = np.load('Data/Glucose_sim_data003.npy', allow_pickle=True)
# data3 = min_maxer.fit_transform(data3)
# x3 = data3[:-1, :]
# y3 = data3[1:, 2:]
# y3_tensor = torch.tensor(y3, dtype=torch.float32)

import torch.nn as nn

citeration = nn.MSELoss()
"""线性回归"""
from sklearn.linear_model import LinearRegression

WX_b = LinearRegression()
rf = WX_b.fit(x1, y1)
y1_pred = rf.predict(x1)
y2_pred = rf.predict(x2)
# y3_pred = rf.predict(x3)
y1_pred_tensor = torch.tensor(y1_pred, dtype=torch.float32)
y2_pred_tensor = torch.tensor(y2_pred, dtype=torch.float32)
# y3_pred_tensor = torch.tensor(y3_pred, dtype=torch.float32)
print('线性回归:', citeration(y1_tensor, y1_pred_tensor).item(),
      citeration(y2_tensor, y2_pred_tensor).item())
# citeration(y3_tensor, y3_pred_tensor))


"""CART"""
from sklearn.tree import DecisionTreeRegressor

cart = DecisionTreeRegressor()
cart_rf = cart.fit(x1, y1)
y1_pred = cart_rf.predict(x1)
y2_pred = cart_rf.predict(x2)
# y3_pred = cart_rf.predict(x3)
y1_pred_tensor = torch.tensor(y1_pred, dtype=torch.float32)
y2_pred_tensor = torch.tensor(y2_pred, dtype=torch.float32)
# y3_pred_tensor = torch.tensor(y3_pred, dtype=torch.float32)
print('CART:', citeration(y1_tensor, y1_pred_tensor).item(),
      citeration(y2_tensor, y2_pred_tensor).item())
# citeration(y3_tensor, y3_pred_tensor))


"""MLP"""
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor()
mlper = mlp.fit(x1, y1)
y1_pred = mlper.predict(x1)
y2_pred = mlper.predict(x2)
y1_pred_tensor = torch.tensor(y1_pred, dtype=torch.float32)
y2_pred_tensor = torch.tensor(y2_pred, dtype=torch.float32)
print('MLP:', citeration(y1_tensor, y1_pred_tensor).item(),
      citeration(y2_tensor, y2_pred_tensor).item())
