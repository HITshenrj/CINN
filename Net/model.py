from abc import ABC
import torch
import torch.nn as nn
import numpy as np

from .couple_layers import RealNVP


class Invertible_Inference(nn.Module, ABC):
    def __init__(self, hidden_size_alpha, hidden_size_a, hidden_size_b, hidden_size_c):
        super(Invertible_Inference, self).__init__()

        self.split_alpha = RealNVP(2, 8, hidden_size_alpha, layers=4)
        self.a12_b34 = RealNVP(1, 1, hidden_size_a, layers=2)
        self.b14_y58 = RealNVP(2, 2, hidden_size_b, layers=2)
        self.c12_y34 = RealNVP(1, 1, hidden_size_c, layers=2)

        # self.alpha = nn.Parameter(torch.ones(4, 2))
        # self._initialize()

    def _initialize(self):
        # 初始化alpha为均匀分布
        self.alpha.data = self.alpha / torch.sum(self.alpha, dim=0)

    def forward(self, X, Z):
        """forward process"""
        X12, X34, X58 = X[:, :2], X[:, 2:-4], X[:, -4:]
        batch_size, Z_dim = Z.size()

        Z_pad = torch.zeros(batch_size, 3 * Z_dim).cuda()
        Z = torch.cat((Z, Z_pad), dim=-1)
        bias = self.split_alpha(Z, invert=False)
        bias1, bias2, bias3, bias4 = bias.view(batch_size, 4, Z_dim).transpose(1, 0)

        a12 = X12 + bias1

        b12 = a12 + bias2
        b34 = self.a12_b34(a12, invert=False) + X34
        b14 = torch.cat((b12, b34), dim=-1)

        c12 = b12 + bias3

        Y12 = c12 + bias4
        Y34 = self.c12_y34(c12, invert=False) + b34
        Y58 = self.b14_y58(b14, invert=False) + X58

        Y = torch.cat((Y12, Y34, Y58), dim=-1)

        return Y

    def back_ward(self, X, Y):
        """backward process to inference variable Z"""
        X12, X34, X58 = X[:, :2], X[:, 2:-4], X[:, -4:]
        Y12, Y34, Y58 = Y[:, :2], Y[:, 2:-4], Y[:, -4:]

        # b
        b14 = self.b14_y58(Y58 - X58, invert=True)
        b12, b34 = b14[:, :2], b14[:, -2:]

        # a
        a12 = self.a12_b34(b34 - X34, invert=True)

        # c
        c12 = self.c12_y34(Y34 - b34, invert=True)

        # bias (batch_size, 2)
        bias1 = a12 - X12
        bias2 = b12 - a12
        bias3 = c12 - b12
        bias4 = Y12 - c12

        bias = torch.cat((bias1, bias2, bias3, bias4), dim=-1)  # (batch_size, 4 * Z_dim)

        c = self.split_alpha(bias, invert=True)

        c = c[:, :2]

        return c


if __name__ == '__main__':
    a = Invertible_Inference(2, 3, 4)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float)
    y = torch.tensor([[ 2.0000,  3.0000,  8.5482, 11.8958,  7.5000,  9.5001, 15.6656, 22.2521],
                        [ 4.0000,  5.0000, 14.7620, 15.7384, 10.0000, 12.0000, 21.1240, 28.3554]], dtype=torch.float)
    z = torch.tensor([[1, 1], [2, 2]], dtype=torch.float)

    out = a.back_ward(x, y)
    print(out)
