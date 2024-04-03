class DefaultConfig(object):
    data_path = 'Data/data.npy'

    epochs = 50000
    batch_size = 1

    hidden_size_alpha = 8
    hidden_size_a = 32
    hidden_size_b = 64
    hidden_size_c = 32

    lr = 4e-3
    weight_decay = 1e-4
    load_model_path = 'ckpt/model.pth'

    alpha = 1.  # 前向回归loss权重
    beta = 2  # 反向loss权重
    sigma = 1e6  # 前向MMD_loss权重


class Glucose_sim_Config(object):
    data_path = 'Data/Glucose_sim_data002.npy'

    epochs = 100000
    batch_size = 3000

    hidden_size_alpha = 512
    hidden_size_2 = 128
    hidden_size_4 = 256

    lr = 5e-4
    weight_decay = 1e-3
    load_model_path = 'ckpt/glucose_model_exp1.pth'

    alpha = 2.  # 前向回归loss权重
    beta = 1.5  # 反向loss权重
    sigma = 2  # 前向MMD_loss权重


class Glucose_sim_Config2(object):
    data_path = 'Data/Glucose_sim_data.npy'

    epochs = 200000
    batch_size = 3000

    hidden_size_1 = 256
    hidden_size_2 = 512
    hidden_size_3 = 512

    lr = 5e-3
    weight_decay = 1e-5
    load_model_path = 'ckpt/glucose_model2.pth'

    alpha = 1.  # 前向回归loss权重
    beta = 5  # 反向loss权重
    sigma = 200.  # 前向MMD_loss权重

class Glucose_sim_Config3(object):
    data_path0 = 'Data/Glucose_sim_data004_0.npy'
    data_path1 = 'Data/Glucose_sim_data004_1.npy'
    data_path2 = 'Data/Glucose_sim_data004_2.npy'
    data_path3 = 'Data/Glucose_sim_data004_3.npy'
    data_path4 = 'Data/Glucose_sim_data004_4.npy'
    data_path5 = 'Data/Glucose_sim_data004_5.npy'
    data_path6 = 'Data/Glucose_sim_data004_6.npy'
    data_path7 = 'Data/Glucose_sim_data004_7.npy'
    data_path8 = 'Data/Glucose_sim_data004_8.npy'
    data_path9 = 'Data/Glucose_sim_data004_9.npy'



    epochs = 100000
    batch_size = 3000

    hidden_size_alpha = 512
    hidden_size_2 = 128
    hidden_size_4 = 256

    lr = 2e-4
    weight_decay = 1e-3
    load_model_path = 'ckpt/glucose_model_change_isu.pth'

    alpha = 2.  # 前向回归loss权重
    beta = 1.5  # 反向loss权重
    sigma = 2  # 前向MMD_loss权重


class Glucose_sim_Config4(object):
    data_path0 = 'usual_generator/data0.npy'
    data_path1 = 'usual_generator/data1.npy'
    data_path2 = 'usual_generator/data2.npy'
    data_path3 = 'usual_generator/data3.npy'
    data_path4 = 'usual_generator/data4.npy'
    data_path5 = 'usual_generator/data5.npy'
    data_path6 = 'usual_generator/data6.npy'
    data_path7 = 'usual_generator/data7.npy'
    data_path8 = 'usual_generator/data8.npy'
    data_path9 = 'usual_generator/data9.npy'



    epochs = 100000
    batch_size = 3000

    hidden_size_alpha = 512
    hidden_size_2 = 128
    hidden_size_4 = 256

    lr = 2e-4  #
    weight_decay = 1e-3
    load_model_path = 'ckpt/other_simulator/model.pth'

    alpha = .5  # 前向回归loss权重
    beta = .5  # 反向loss权重
    sigma = 2  # 前向MMD_loss权重