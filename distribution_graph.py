import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def distribution_2D():
    np.set_printoptions(threshold=np.inf)
    plt.figure()
    color_list = ['grey','brown','red','sandybrown','orange','yellow','lawngreen','cyan','blue','pink']

    for i in range(10):
        data = np.load('Data\Glucose_sim_data004_{}.npy'.format(i))
        print(data.shape)
        standarder = StandardScaler()
        data = standarder.fit_transform(data)
        pca = PCA(n_components=2)
        result = pca.fit_transform(data)
        print(result.shape)
        plt.scatter(
            result[360:380,0],
            result[360:380,1],
            # result[370:540,0],
            # result[370:540,1],
            c=color_list[i],
            label='distribution{}'.format(i),
            s = 5
        )
    plt.legend()
    plt.show()

def distribution_3D():
    np.set_printoptions(threshold=np.inf)
    fig = plt.figure()
    color_list = ['grey','brown','red','sandybrown','orange','yellow','lawngreen','cyan','blue','pink']
    ax = fig.gca(projection="3d")

    for i in range(10):
        data = np.load('Data\Glucose_sim_data004_{}.npy'.format(i))
        print(data.shape)
        pca = PCA(n_components=3)
        result = pca.fit_transform(data)
        print(result.shape)
        ax.scatter(
            result[380:390, 0],
            result[380:390, 1],
            result[380:390, 2],
            # result[370:540,0],
            # result[370:540,1],
            # result[370:540,2],
            c=color_list[i],
            label='distribution{}'.format(i),
            s = 15
        )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

distribution_2D()
# distribution_3D()