from abc import ABC
import torch
import torch.nn as nn
import numpy as np

from .couple_layers import RealNVP
from .OWN import OWNNorm


class Insulin_CHO_sim(nn.Module, ABC):
    def __init__(self, hidden_size_alpha, hidden_size_2, hidden_size_4):
        """
            b -> e: x0, x10, x1, x11对x4, x8 x3, x12的控制
            a -> c -> f: x0, x10, x1, x11, x2, x5对x4, x8 x3, x12的控制
            d -> h: x0, x10, x1, x11对x2, x5的控制
            g: x0, x10对x1, x11的控制
        """
        super(Insulin_CHO_sim, self).__init__()

        self.alpha1 = nn.Parameter(torch.randn(2, 10))
        self.alpha1_bias = nn.Parameter(torch.randn(10, ))
        self.alpha_inverse_transform = RealNVP(5, 5, hidden_size_alpha, layers=8)
        # self.alpha2 = nn.Parameter(torch.randn(10, 10))
        # self.alpha2_bias = nn.Parameter(torch.randn(10, ))
        self.own_ = OWNNorm(norm_groups=1)

        # self.split_alpha = RealNVP(2, 8, hidden_size_alpha, layers=6)
        self.a_c = RealNVP(1, 1, hidden_size_2, layers=2)
        self.b_e = RealNVP(1, 1, hidden_size_2, layers=2)
        self.c_f = RealNVP(1, 1, hidden_size_2, layers=2)
        self.ef_710 = RealNVP(2, 2, hidden_size_4, layers=2)
        self.d_h = RealNVP(1, 1, hidden_size_2, layers=2)
        self.h_x56 = RealNVP(1, 1, hidden_size_2, layers=2)
        self.g_x34 = RealNVP(1, 1, hidden_size_2, layers=2)

    def orthogonality_weight_normalization(self):
        """正交权重重参数化"""
        self.own_(self.alpha1)

    def forward(self, X, Z, own=True):
        """forward process"""
        X12, X34, X56, X710 = X[:, :2], X[:, 2:4], X[:, 4:6], X[:, -4:]
        batch_size, Z_dim = Z.size()

        # Z_pad = torch.zeros(batch_size, 4 * Z_dim).cuda()
        # Z = torch.cat((Z, Z_pad), dim=-1)
        # bias = self.split_alpha(Z, invert=False)
        # bias1, bias2, bias3, bias4, bias5 = bias.view(batch_size, 5, Z_dim).transpose(1, 0)
        # 每次前向都做正交重参数化
        if own:
            self.alpha1.data = self.own_(self.alpha1).view(self.alpha1.data.size(0), -1)
        # self.alpha2.data = self.own_(self.alpha2).view(self.alpha2.data.size(0), -1)

        bias = torch.matmul(Z, self.alpha1) + self.alpha1_bias
        # bias = torch.matmul(bias, self.alpha2) + self.alpha2_bias
        bias = self.alpha_inverse_transform(bias, invert=False)
        bias1, bias2, bias3, bias4, bias5 = bias.view(batch_size, 5, -1).transpose(1, 0)

        a = X12 + bias1
        b = a + bias2
        c = X34 + self.a_c(a, invert=False)

        d = b + bias3
        e = c + self.b_e(b, invert=False)
        f = X56 + self.c_f(c, invert=False)
        ef = torch.cat((e, f), dim=-1)
        g = d + bias4
        h = e + self.d_h(d, invert=False)

        Y12 = g + bias5
        Y34 = h + self.g_x34(g, invert=False)
        Y56 = f + self.h_x56(h, invert=False)
        Y710 = X710 + self.ef_710(ef, invert=False)

        Y = torch.cat((Y12, Y34, Y56, Y710), dim=-1)

        return Y

    def back_ward(self, X, Y):
        """backward process to inference Variable Z"""
        self.alpha1.data = self.own_(self.alpha1).view(self.alpha1.data.size(0), -1)
        X12, X34, X56, X710 = X[:, :2], X[:, 2:4], X[:, 4:6], X[:, -4:]
        Y12, Y34, Y56, Y710 = Y[:, :2], Y[:, 2:4], Y[:, 4:6], Y[:, -4:]

        ef = self.ef_710(Y710 - X710, invert=True)
        e, f = ef[:, :2], ef[:, -2:]
        c, h = self.c_f(f - X56, invert=True), self.h_x56(Y56 - f, invert=True)

        a = self.a_c(c - X34, invert=True)
        b = self.b_e(e - c, invert=True)
        d = self.d_h(h - e, invert=True)
        g = self.g_x34(Y34 - h, invert=True)

        bias = torch.cat((a - X12, b - a, d - b, g - d, Y12 - g), dim=-1)
        bias = self.alpha_inverse_transform(bias, invert=True)
        # out = torch.matmul((bias - self.alpha2_bias), self.alpha2.T)
        out = torch.matmul((bias - self.alpha1_bias), self.alpha1.T)
        # out = torch.matmul(out1, torch.inverse(torch.matmul(self.alpha1, self.alpha1.T)))

        return out


if __name__ == '__main__':
    model = Insulin_CHO_sim(100, 256, 512)
    z = torch.tensor([[100000., 200000.]], dtype=torch.float32)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32)
    y = model(x, z)
    print(torch.matmul(model.alpha1, model.alpha1.T))
    print(y)
    z_ = model.back_ward(x, y)
    print(z_)
    # print(model.alpha1, torch.inverse(model.alpha1))
