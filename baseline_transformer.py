import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from baseline.encoders import TransformerEncoder
from config import Glucose_sim_Config3


def generate_data(n, opt):
    dic = {0: opt.data_path0,
           1: opt.data_path1,
           2: opt.data_path2,
           3: opt.data_path3,
           4: opt.data_path4,
           5: opt.data_path5,
           6: opt.data_path6,
           7: opt.data_path7,
           8: opt.data_path8,
           9: opt.data_path9
           }

    data = np.load(dic[n], allow_pickle=True)

    data_ = []
    bias = [1, 1, 0.001, 0.01, 0.001, 1, 0.1, 0.001, 0.1, 0.1, 0.1, 1, 0.1, 0.1]
    for i in range(data.shape[0]):
        if i % 480 == 0 and i >= 960 and i < data.shape[0] - 480:
            d = np.array([data[j, :] for j in range(i, i + 71)]) * bias
            d_ = np.c_[d[:-1, :], d[1:, :]]
            data_.append(d_)

    data = np.array([l for s in data_ for l in s])

    np.random.shuffle(data)

    data_tensor = torch.from_numpy(data).to(dtype=torch.float32).unsqueeze(0)
    return data_tensor[:, :, :-14], data_tensor[:, :, :-14], data_tensor

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=26, out_features=64)
        self.hidden2 = nn.Linear(in_features=64, out_features=14)

    def forward(self,input):
        output = self.hidden1(input)
        output = self.relu(output)
        output = self.hidden2(output)
        output = self.relu(output)
        return output


class baseline_MLP(nn.Module):
    ENCODER_HIDDEN_DIM = 1024
    def __init__(self, input_dim, embed_dim=256,
                 encoder_name='transformer',
                 encoder_blocks=3,
                 encoder_heads=8,
                 device=None) -> None:


        super(baseline_MLP, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder_blocks = encoder_blocks
        self.encoder_heads = encoder_heads
        self.encoder_name = encoder_name
        self.device = device

        self.encoder = TransformerEncoder(input_dim=self.input_dim,
                                          embed_dim=self.embed_dim,
                                          hidden_dim=self.ENCODER_HIDDEN_DIM,
                                          heads=self.encoder_heads,
                                          blocks=self.encoder_blocks,
                                          device=self.device)

        self.mlp= MLP()

    def forward(self,input):
        output = self.encoder(input).squeeze().t()
        output = self.mlp(output)
        return output



def main(opt, adj_matrix, Ux: list):
    tb_logger = SummaryWriter('./log_baselines')
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    model = baseline_MLP(input_dim=32,embed_dim=32)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)  # , weight_decay=opt.weight_decay)
    optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    X0, Y0, xy0 = generate_data(0, opt)
    X1, Y1, xy1 = generate_data(1, opt)
    # X2, Y2, xy2 = generate_data(2, opt)
    # X3, Y3, xy3 = generate_data(3, opt)
    # X4, Y4, xy4 = generate_data(4, opt)
    # X5, Y5, xy5 = generate_data(5, opt)
    # X6, Y6, xy6 = generate_data(6, opt)
    # X7, Y7, xy7 = generate_data(7, opt)
    # X8, Y8, xy8 = generate_data(8, opt)
    # X9, Y9, xy9 = generate_data(9, opt)

    train_X_1 = X0[:,:60000, :]
    train_X_2 = Y0[:,:60000, :]
    train_Y = xy0[:,:60000, :]
    test_X_1 = X0[:,:60000:, :]
    test_X_2 = Y0[:,:60000:, :]
    test_Y = xy0[:,:60000:, :]
    print(train_X_2.shape)
    print(test_X_2.shape)
    flag = False
    count = 0
    delta_loss = 0.0
    for epoch in range(opt.epochs):
        # print('\n=====epoch {0} ====='.format(epoch + 1))
        model.train()
        loss1 = 0.0
        loss2 = 0.0

        for step in range(0, len(train_Y), opt.batch_size):
            # print(torch.matmul(model.alpha1, model.alpha1.T))
            input_X1 = train_X_1[step:step + opt.batch_size]
            input_X2 = train_X_2[step:step + opt.batch_size]
            input_Y = train_Y[step:step + opt.batch_size]
            if torch.cuda.is_available():
                input_X1 = input_X1.cuda()
                input_X2 = input_X2.cuda()
                input_Y = input_Y.cuda()

            forward_out = model(input_X1)
            forward_loss = criterion(forward_out[:, 2:], input_Y[:, 2:]) * opt.alpha
            loss1 += forward_loss.item()

            backward_out = model(input_X2)
            backward_loss = criterion(backward_out[:, :2], input_Y[:, :2]) * opt.beta
            loss2 += backward_loss.item()

            loss = forward_loss + backward_loss
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

        if epoch == 20000:
            torch.save(model.state_dict(), opt.load_model_path)
            break

        print('\n=====epoch {0} finished=====\nepoch_forward_loss = {1:.6f}\nepoch_inverse_loss = {2:.6f}'
              .format(epoch + 1, loss1, loss2))
        tb_logger.add_scalars('./log_exp_2/' + 'loss_train_forward', {'forward_loss': loss1}, epoch)
        tb_logger.add_scalars('./log_exp_2/' + 'loss_train_inverse', {'inverse_loss': loss2}, epoch)

        for_loss, invert_loss = invertible_Inference(opt, model, (test_X_1, test_X_2, test_Y))
        print('=====0测试正向loss{0:.4f}逆向loss:{1:.4f}====='.format(for_loss, invert_loss))
        tb_logger.add_scalars('./log_exp_2/' + 'loss_test_forward_1', {'forward_loss': for_loss}, epoch)
        tb_logger.add_scalars('./log_exp_2/' + 'loss_test_inverse_1', {'inverse_loss': invert_loss}, epoch)
        for i in range(1, 2):
            locals()["for_loss" + str(i)], locals()["invert_loss" + str(i)] = invertible_Inference(opt, model, (
            locals()["X" + str(i)], locals()["Y" + str(i)], locals()["xy" + str(i)]))
            print('====={0:d}测试正向loss{1:.4f}逆向loss:{2:.4f}====='.format(i, locals()["for_loss" + str(i)],
                                                                        locals()["invert_loss" + str(i)]))
            tb_logger.add_scalars('./log_exp_2/' + 'loss_test_forward_{}'.format(i),
                                  {'forward_loss': locals()["for_loss" + str(i)]}, epoch)
            tb_logger.add_scalars('./log_exp_2/' + 'loss_test_inverse_{}'.format(i),
                                  {'inverse_loss': locals()["invert_loss" + str(i)]}, epoch)


def invertible_Inference(opt, model, test_data):
    """反向推断用于评估"""
    # model.orthogonality_weight_normalization()
    model.eval()
    criterion = nn.MSELoss()
    test_X_1, test_X_2, test_Y = test_data
    forward_loss = 0.0
    invert_loss = 0.0

    with torch.no_grad():
        for step in range(0, len(test_Y), opt.batch_size):
            input_X1 = test_X_1[step:step + opt.batch_size]
            input_X2 = test_X_2[step:step + opt.batch_size]
            input_Y = test_Y[step:step + opt.batch_size]
            if torch.cuda.is_available():
                input_X1 = input_X1.cuda()
                input_X2 = input_X2.cuda()
                input_Y = input_Y.cuda()
            forward_out = model(input_X1)
            backward_out = model(input_X2)
            forward_loss += criterion(forward_out[:, 2:], input_Y[:, 2:]).item()
            invert_loss += criterion(backward_out[:, :2], input_Y[:, :2]).item()

    return forward_loss, invert_loss


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
    opt = Glucose_sim_Config3()
    main(opt, adj_matrix, Ux)
