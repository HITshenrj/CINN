import copy
import numpy as np


class CausalGraph:
    def __init__(self,
                 train=True,
                 permute=True,
                 intervene_idx=None):
        """
        Create the causal graph structure.
        :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
        """
        self.final = []
        self.nodes = [CausalNode(i) for i in range(16)]
        self.nodes[0].val = 0
        self.nodes[1].val = 1.0
        self.nodes[2].val = 1.0
        self.nodes[3].val = 1.0
        self.nodes[4].val = 1.0
        self.nodes[5].val = 1.0
        self.nodes[6].val = 1.0
        self.nodes[7].val = 1.0
        self.nodes[8].val = 1.0
        self.nodes[9].val = 1.0
        self.nodes[10].val = 1.0
        self.nodes[11].val = 1.0
        self.nodes[12].val = 1.0
        self.nodes[13].val = 1.0
        self.nodes[14].val = 1.0
        self.nodes[15].val = 1.0

    def calculate(self,k):
        """
        干预完了之后计算没干预点的值
        :return:
        """
        delta = np.zeros(16)
        delta[0] = k * 0.25 * np.random.randn()
        delta[1] = self.nodes[0].val * 0.01
        delta[2] = 0.02 * self.nodes[1].val
        delta[3] = 0.02 * self.nodes[1].val
        delta[4] = 0.01 * self.nodes[3].val
        delta[5] = -0.01 * self.nodes[2].val
        delta[6] = -0.02 * self.nodes[4].val - 0.01 * self.nodes[6].val
        delta[7] = 0.01 * self.nodes[6].val
        delta[8] = 0.03 * self.nodes[7].val
        delta[9] = -0.03 * self.nodes[5].val
        delta[10] = 0.01 * self.nodes[9].val - 0.01 * self.nodes[10].val
        delta[11] = 0.01 * self.nodes[10].val
        delta[12] = -0.03 * self.nodes[11].val
        delta[13] = 0.01 * self.nodes[8].val - 0.01 * self.nodes[12].val
        delta[14] = 0.02 * self.nodes[13].val + 0.03 * self.nodes[14].val
        delta[15] = 0.01 * self.nodes[14].val

        for idx in range(len(self.nodes)):
            if idx == 0:
                self.nodes[0].val = delta[0]
            else:
                if not self.nodes[idx].intervened:
                    self.nodes[idx].last_val = copy.deepcopy(self.nodes[idx].val)
                self.nodes[idx].val = self.nodes[idx].val + delta[idx]

    def sample_all(self):
        """
        Sample all nodes according to their causal relations
        :return: sampled_vals (np.ndarray) array of sampled values
        """
        sampled_vals = np.zeros(16)
        for idx in range(len(self.nodes)):
            sampled_vals[idx] = copy.deepcopy(self.nodes[idx].val)
            if idx == 15:
                # print(sampled_vals)
                self.final.append(sampled_vals)
        return sampled_vals

    def get_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.nodes[node_idx].val

    def get_last_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.nodes[node_idx].last_val

    def clear(self):
        self.nodes[0].val = 0
        self.nodes[1].val = 1.0
        self.nodes[2].val = 1.0
        self.nodes[3].val = 1.0
        self.nodes[4].val = 1.0
        self.nodes[5].val = 1.0
        self.nodes[6].val = 1.0
        self.nodes[7].val = 1.0
        self.nodes[8].val = 1.0
        self.nodes[9].val = 1.0
        self.nodes[10].val = 1.0
        self.nodes[11].val = 1.0
        self.nodes[12].val = 1.0
        self.nodes[13].val = 1.0
        self.nodes[14].val = 1.0
        self.nodes[15].val = 1.0

class CausalNode:
    def __init__(self, idx):
        """
        Create data structure for node which knows its parents
        :param idx: index of node in graph
        :param adj_mat: upper triangular matrix for graph
        """
        self.id = idx
        self.val = None
        self.last_val = None
        self.intervened = False

    def increase(self, val):
        self.last_val = copy.deepcopy(self.val)
        self.val = val + self.val
        self.intervened = True

for z in range(1,11):
    A = CausalGraph()
    for m in range(20):
        for i in range(400):
            A.calculate(z)
            A.sample_all()
        A.clear()

    A.final = np.array(A.final)
    np.save('./data{}.npy'.format(z-1), A.final)