from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .couple_layers import RealNVP


class Insulin_CHO_sim(nn.Module, ABC):
    def __init__(self, hidden_size1, hidden_size2, hidden_size3):
        super(Insulin_CHO_sim, self).__init__()

        self.CPS1 = RealNVP(2, 2, hidden_size1, layers=2)
        self.CPS2 = RealNVP(6, 2, hidden_size2, layers=2)
        self.CPS3 = RealNVP(8, 8, hidden_size3, layers=2)
        self.Alpha = nn.Parameter(torch.randn(3, 2))

    def forward(self, X, Z):
        """forward process"""
        batch_size, Z_dim = Z.size()
        # 归一化
        self.Alpha.data = F.softmax(self.Alpha, dim=-1)

        X1, X2, X3 = X[:, :2], X[:, 2:6], X[:, 6:]
        Z = Z.unsqueeze(1).expand(batch_size, 2, 2)

        alpha1 = self.Alpha[0].unsqueeze(1).expand(batch_size, 2, 2)
        Z1, Z2 = (Z * alpha1).transpose(1, 0)

        I1 = torch.cat((Z1, X1 + Z2), dim=-1)
        O1 = self.CPS1(I1, invert=False)

        alpha2 = self.Alpha[1].unsqueeze(1).expand(batch_size, 2, 4)
        O1 = O1.unsqueeze(1).expand(batch_size, 2, 4)
        O11, O12 = (alpha2 * O1).transpose(1, 0)

        I2 = torch.cat((O11, X2 + O12), dim=-1)
        O3 = self.CPS2(I2, invert=False)

        alpha3 = self.Alpha[2].unsqueeze(1).expand(batch_size, 2, 8)
        O3 = O3.unsqueeze(1).expand(batch_size, 2, 8)
        O31, O32 = (alpha3 * O3).transpose(1, 0)

        I3 = torch.cat((O31, X3 + O32), dim=-1)

        out = self.CPS3(I3, invert=False)

        return out

    def back_ward(self, X, Y):
        """backward process to inference Variable Z"""
        batch_size = X.size(0)
        X1, X2, X3 = X[:, :2], X[:, 2:6], X[:, 6:]

        O31, O32 = self.CPS3(Y, invert=True).view(batch_size, 2, 8).transpose(1, 0)
        O3 = O31 + O32 - X3

        O21, O22 = self.CPS2(O3, invert=True).view(batch_size, 2, 4).transpose(1, 0)
        O2 = O21 + O22 - X2

        O11, O12 = self.CPS1(O2, invert=True).view(batch_size, 2, 2).transpose(1, 0)

        Z = O11 + O12 - X1

        return Z


if __name__ == '__main__':
    x = torch.randn(128, 14)
    y = torch.randn(128, 16)
    z = torch.randn(128, 2)

    model = Insulin_CHO_sim(32, 64, 128)
    model.back_ward(x,y)