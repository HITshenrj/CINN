from G2M_model import Graph2Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import Glucose_sim_Config3
import argparse

def LN(n):
    data_tensor = copy.deepcopy(n)
    mean = data_tensor.mean(-1)
    var = (data_tensor - mean.reshape(-1, 1)).pow(2).mean(-1)
    data_tensor = (data_tensor - mean.reshape(-1, 1)) / var.sqrt().reshape(-1, 1)
    return data_tensor, mean, var

def main(adj_matrix, Ux: list):

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--layers', type=int, default=2)
    arg_parser.add_argument('--lr', type=float, default=0.0001)
    arg_parser.add_argument('--fn', type=int, nargs='+', default=['32', '32'])
    arg_parser.add_argument('--bn', type=int, nargs='+', default=['32', '32'])
    arg_parser.add_argument('--decade_epoch', type=int, default=1000)
    arg_parser.add_argument('--gamma', type=float, default=0.5)
    arg_parser.add_argument('--hidden', type=int, nargs='+', default=['16', '64'])
    arg_parser.add_argument('--log_dir', type=str, default='./log/lr_1e-4_f_32_h_32')
    arg_parser.add_argument('--weight', type=float, default=5.)
    arg_parser.add_argument('--reload', type=bool, default=False)

    args = arg_parser.parse_args()
    args.fn = [int(fn) for fn in args.fn]
    args.bn = [int(bn) for bn in args.bn]
    args.hidden = [int(hidden) for hidden in args.hidden]

    new_model = Graph2Model(adj_matrix, Ux, args.layers, args.fn, args.bn, args.hidden, args.weight)
    new_model.load_state_dict(torch.load("./ckpt/lr_1e-4_f_32_h_16_1500.pth",map_location='cpu'))
    data1 = np.load(Glucose_sim_Config3.data_path0, allow_pickle=True)
    data_ = []
    bias = [1, 1, 0.001, 0.01, 0.001, 0.1, 0.001, 0.001, 0.001, 0.001, 0.1, 0.1, 0.1, 0.001]
    for i in range(data1.shape[0]):
        if i % 180 == 0 and i != 0:
            d = np.array([data1[j, :] for j in range(i, i + 20)]) * bias
            d_ = np.c_[d[:-1, :], d[1:, 2:]]
            data_.append(d_)
    data = np.array([l for s in data_ for l in s])

    data_tensor = torch.from_numpy(data).to(dtype=torch.float32)

    LN_at_St, _, _ = LN(data_tensor[:, :-12])  # 正向输入
    LN_St1, _, _, = LN(data_tensor[:, -12:])  # 反向输入
    LN_St, mean, var = LN(data_tensor[:, 2:-12])  # 第一项用于反向计算的输入，其余两项用于正向算loss
    # print(LN_St.shape)
    LN_St = torch.cat([torch.zeros([LN_St.shape[0], 2]), LN_St], dim=-1)

    with open('./same_distribution.txt','w') as f:
        for i in range(3000,3020):
            input_X_f = LN_at_St[i,:].reshape(1,-1)
            input_X_b = LN_St1[i,:].reshape(1,-1)
            input_Y=LN_St[i,:].reshape(1,-1)
            forward = data_tensor[i,-12:].reshape(1,-1)
            backward = data_tensor[i,:-12].reshape(1,-1)
            f.write('---预测结果---\n')
            a = new_model.forward(input_X_f)
            pred_In, true_In = new_model.back_ward(input_X_b, input_Y,backward)
            f.write('\n正向真实值:{}\n正向预测值：{}\n'.format(forward,a))
            f.write('\n反向真实值:{}\n反向预测值：{}\n'.format(true_In,pred_In))

    # with open('./different_distribution.txt', 'w') as f:
    #     for i in range(3000, 3020):
    #         inputX = xy10[i, :-12].reshape(1, -1)
    #         inputY = xy10[i, -12:].reshape(1, -1)
    #         f.write('---预测结果---\n')
    #         f.write(str(inputX))
    #         a = new_model.forward(inputX)
    #         pred_In, true_In = new_model.back_ward(inputX, inputY)
    #         f.write('\n正向真实值:{}\n正向预测值：{}\n'.format(inputY, a))
    #         f.write('\n反向真实值:{}\n反向预测值：{}\n'.format(true_In, pred_In))


def test_data():
    standarder = StandardScaler()
    np.set_printoptions(threshold=np.inf)
    data1 = np.load(Glucose_sim_Config3.data_path0, allow_pickle=True)

    data1_1 = torch.from_numpy(data1[1:, 0:2]).to(dtype=torch.float32)
    data1_2 = torch.from_numpy(data1[:-1, 2:]).to(dtype=torch.float32)
    data1 = torch.cat((data1_1, data1_2), dim=-1).numpy()

    data1 = standarder.fit_transform(data1)
    print(data1[6599,:])
    data1_tensor = copy.deepcopy(torch.from_numpy(data1)).to(dtype=torch.float32)

    data10 = np.load(Glucose_sim_Config3.data_path9, allow_pickle=True)
    data10_1 = torch.from_numpy(data10[1:, 0:2]).to(dtype=torch.float32)
    data10_2 = torch.from_numpy(data10[:-1, 2:]).to(dtype=torch.float32)
    data10 = torch.cat((data10_1, data10_2), dim=-1).numpy()
    standarder10 = StandardScaler()
    data10 = standarder10.fit_transform(data10)
    print(data10[6599,:])
    data10_tensor = copy.deepcopy(torch.from_numpy(data10)).to(dtype=torch.float32)

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
    #test_data()