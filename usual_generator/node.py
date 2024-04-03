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
        self.nodes = [CausalNode(i) for i in range(5)]
        self.nodes[0].val = 1.0
        self.nodes[1].val = 1.0
        self.nodes[2].val = 1.0
        self.nodes[3].val = 1.0
        self.nodes[4].val = 1.0

    def calculate(self,k):
        """
        干预完了之后计算没干预点的值
        :return:
        """
        delta = np.zeros(5)
        delta[1] = - self.nodes[2].val * 0.1 + self.nodes[0].val * 0.5
        delta[0] = - self.nodes[0].val * 0.1  - 1.0   # 自衰减
        delta[2] = 0.1 * self.nodes[3].val - 0.01 * self.nodes[2].val
        delta[3] = 0.1 * self.nodes[4].val - 0.01 * self.nodes[3].val
        delta[4] = k * np.random.randn()

        for idx in range(len(self.nodes)):
            # if not self.nodes[idx].intervened:
            #     self.nodes[idx].last_val = copy.deepcopy(self.nodes[idx].val)
            if idx == 4:
                self.nodes[4].val = delta[4]
            else:
                self.nodes[idx].val = self.nodes[idx].val + delta[idx]

    def sample_all(self):
        """
        Sample all nodes according to their causal relations
        :return: sampled_vals (np.ndarray) array of sampled values
        """
        sampled_vals = np.zeros(5)
        for idx in range(len(self.nodes)):
            sampled_vals[idx] = copy.deepcopy(self.nodes[idx].val)
            if idx == 4:
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

for z in  range(10):
    A = CausalGraph()
    for i in range(8000):
        A.calculate(z)
        A.sample_all()
    A.final = np.array(A.final)
    np.save('./data{}.npy'.format(z), A.final)