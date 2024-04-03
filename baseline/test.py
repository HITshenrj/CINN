from MLP import MLP
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import Glucose_sim_Config3


def main(adj_matrix,Ux):

    new_model = MLP(adj_matrix, Ux)
    new_model.load_state_dict(torch.load("../ckpt/glucose_model_change_isu.pth",map_location='cpu'))
    data1 = np.load('../'+Glucose_sim_Config3.data_path3, allow_pickle=True)
    data_ = []
    for i in range(data1.shape[0]):
        if i % 180 == 0 and i != 0:
            d = np.array([data1[j, :] for j in range(i, i + 20)])
            d_ = np.c_[d[:-1, :], d[1:, 2:]]
            data_.append(d_)
    data = np.array([l for s in data_ for l in s])

    standarder = StandardScaler()
    data = standarder.fit_transform(data)
    data1 = data[:, :2]
    data2 = data[:, -12:]
    data3 = np.c_[data, data1, data2]
    data3[:, :2] = 0
    data4 = np.c_[data, data1, data2]
    data4[:, 14:-14] = 0
    data_tensor_1 = torch.from_numpy(data4).to(dtype=torch.float32)
    data_tensor_2 = torch.from_numpy(data3).to(dtype=torch.float32)



    # with open('./same_distribution.txt','w') as f:
    #     for i in range(3000,3020):
    #         inputX1=data_tensor_1[i,:-14].reshape(1,-1)
    #         inputX2=data_tensor_2[i,:-14].reshape(1,-1)
    #         inputY=data_tensor_2[i,-14:].reshape(1,-1)
    #         f.write('---预测结果---\n')
    #         f.write(str(inputX1))
    #         f.write(str(inputX2))
    #         a = new_model.forward(inputX1)
    #         b = new_model.forward(inputX2)
    #         f.write('\n正向真实值:{}\n正向预测值：{}\n'.format(inputY,a))
    #         f.write('\n反向真实值:{}\n反向预测值：{}\n'.format(inputY,b))

    with open('./differ_distribution.txt','w') as f:
        for i in range(3000,3020):
            inputX1=data_tensor_1[i,:-14].reshape(1,-1)
            inputX2=data_tensor_2[i,:-14].reshape(1,-1)
            inputY=data_tensor_2[i,-14:].reshape(1,-1)
            f.write('---预测结果---\n')
            f.write(str(inputX1))
            f.write(str(inputX2))
            a = new_model.forward(inputX1)
            b = new_model.forward(inputX2)
            f.write('\n正向真实值:{}\n正向预测值：{}\n'.format(inputY,a))
            f.write('\n反向真实值:{}\n反向预测值：{}\n'.format(inputY,b))

if __name__ == '__main__':
    e01 = 0.46122
    e02 = 0.001536032069546028
    e12 = 0.03330367437547978
    e23 = 0.0007833659108679641
    e3_12 = 0.0766
    e43 = 0.087114
    e56 = 0.5063563180709586
    e57 = 0.08446071467598117
    e64 = 0.2951511925794614
    e78 = 0.0046374
    e83 = 0.00469
    e10_5 = 0.0019
    e10_11 = 0.0152
    e11_5 = 0.0078

    #              0  1  2  3  4  5  6  7  8  9  10 11  12
    adj_matrix = [[0, e01, e02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                  [0, 0, e12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
                  [0, 0, 0, e23, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e3_12],  # 3
                  [0, 0, 0, e43, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                  [0, 0, 0, 0, 0, 0, e56, e57, 0, 0, 0, 0, 0],  # 5
                  [0, 0, 0, 0, e64, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
                  [0, 0, 0, 0, 0, 0, 0, 0, e78, 0, 0, 0, 0],  # 7
                  [0, 0, 0, e83, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
                  [0, 0, 0, 0, 0, e10_5, 0, 0, 0, 0, 0, e10_11, 0],  # 10
                  [0, 0, 0, 0, 0, e11_5, 0, 0, 0, 0, 0, 0, 0],  # 11
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 12

    Ux = [0, 10]

    main(adj_matrix,Ux)