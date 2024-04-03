import torch
import numpy as np


def MSE(label_Y, pred_Y):
    """
    :param label_Y: 标签
    :param pred_Y: 预测
    :return:
    """

    batch_size, feature_dim = pred_Y.size()
    # print(pred_Y.size())
    minus = label_Y - pred_Y
    # print(minus)
    loss = torch.mul(minus, minus)
    loss = torch.sum(loss, dim=-1)
    loss /= feature_dim
    loss = torch.sum(loss, dim=0)
    loss /= batch_size

    return loss.cpu().item()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        sum(kernel_val): 多个核矩阵之和
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        loss: MMD loss
    """
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def prepareData(filePath):
    samples_dict = np.load(filePath, allow_pickle=True).item()

    samples_matrix = np.vstack((samples_dict['M1'], samples_dict['M3'], samples_dict['M3'], samples_dict['M4'],
                                samples_dict['M5'], samples_dict['M5'], samples_dict['M7'], samples_dict['M8'],
                                samples_dict['Insulin']))

    samples_train = samples_matrix[:, :]

    return samples_train


def orthogonality_constraint(T):
    """正交性约束"""
    return torch.sum(T[0] * T[1], dim=-1)


def compute_num(l):
    """计算元素个数"""
    total_num = 0
    for _ in l:
        total_num += len(_)
    return total_num


def binary_split(l):
    """按照层次尽可能二分数组"""
    if len(l) == 1:
        return []
    total_num = compute_num(l)
    cur_num = 0
    for i, sub_l in enumerate(l):
        cur_num += len(sub_l)
        if cur_num >= total_num / 2.0:
            if i != len(l) - 1:
                return binary_split(l[:i + 1]) + [i + 1]
            else:
                return [i]


if __name__ == '__main__':
    # a = torch.tensor([[5., 1.], [2., 2.], [0., 0.]])
    # b = torch.tensor([[4.9, 1.5], [1.5, 1.5], [0., 0.]])
    # print(MSE(a, b))
    # criterion = torch.nn.MSELoss()
    # print(criterion(a, b))
    # a = prepareData('Data/data.npy').T
    # print(a.shape)
    # a = torch.tensor([[5., 1.], [2,1]])
    # b = torch.tensor([1, 1])
    # print(orthogonality_constraint(a))
    print(binary_split([[0, 1], [0, 1], [2, 3, 4, 5], [6, 7, 9, 10], [8], [11]]))
