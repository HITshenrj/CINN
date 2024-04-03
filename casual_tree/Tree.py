import copy
import math
import numpy as np
import queue
from .Graph import Graph


class TreeNode(object):
    def __init__(self, name):
        self.name = name
        self.sons = []    #儿子节点

    def add(self, son):
        self.sons.append(son)

    def get_name(self):
        return self.name

    @staticmethod
    def get_children_num(self):
        num = 1
        for s in self.sons:
            num += s.get_children_num()
        return num

    def __str__(self):
        """广搜查看"""
        return str(self.name)


class Tree(object):
    def __init__(self, Ux: list, graph=None):

        self.graph = graph
        self.Ux_node = Ux   # 基于数字的外生变量集合
        self.dis_matrix = None
        self.path_matrix = None
        self._Floyd()
        self.sons = {}   #dic
        self.unconnected = ()
        self.get_longest_trees()
        self.con_Tree = None
        self.sons_node = {}
        self.combine_trees()

    def _Floyd(self):
        """Floyd算法计算最长路径"""
        dis_matrix = copy.deepcopy(self.graph.adjacency_matrix)

        for i in range(len(dis_matrix)):
            for j in range(len(dis_matrix)):
                if dis_matrix[i][j] == 0:
                    dis_matrix[i][j] = math.inf
                else:
                    dis_matrix[i][j] = -math.log(dis_matrix[i][j])
        num = len(dis_matrix)
        path_matrix = np.full((num, num), math.inf)  # 最短路径的最后一个点

        for i in range(num):
            for j in range(num):
                for k in range(num):
                    if j == k:
                        continue
                    if dis_matrix[j][k] > dis_matrix[j][i] + dis_matrix[i][k]:
                        dis_matrix[j][k] = dis_matrix[j][i] + dis_matrix[i][k]
                        path_matrix[j][k] = i
        self.dis_matrix = np.array(dis_matrix)
        self.path_matrix = path_matrix

    def get_longest_trees(self):
        """构建多源最长路径树(默认可控节点不连通)"""
        # 首先排除不连通的点
        connected = set()
        unconnected = set(range(self.graph.get_node_num()))
        for U in self.Ux_node:
            connected = connected.union(set(self.graph.get_connected(U)))
        unconnected = unconnected - connected
        self.unconnected = unconnected

        # 为各个节点找到根
        roots = {}
        root_matrix = np.argmin(self.dis_matrix[self.Ux_node], axis=0)   # 为每个变量寻找最近的外生变量   分树关键代码
        for i, r in enumerate(root_matrix):
            if i in self.Ux_node or i in unconnected:  #本身是外生变量or非连通节点
                roots[i] = -1
            else:
                roots[i] = self.Ux_node[r]    #i的根 外生变量x

        # 为各个节点找到儿子节点
        sons = {-1: self.Ux_node}
        for i in range(self.graph.get_node_num()):
            sons[i] = []
        for U in self.Ux_node:
            path_U = self.path_matrix[U]
            for i, f in enumerate(path_U):
                if roots[i] == U:
                    # 是以U为根
                    if f == math.inf:
                        sons[U].append(i)
                    else:
                        sons[int(f)].append(i)    #floyd保证此时的f是最大点 f->i一定是最后一条边
        self.sons = sons
        print(self.sons)

    def combine_trees(self):
        """合并树"""
        def construct_subtree(son_dic, n):
            Node = TreeNode(n)
            sons = son_dic[n]
            for i in sons:
                Node.sons.append(construct_subtree(son_dic, i))
            self.sons_node[Node.name] = Node
            return Node

        self.con_Tree = construct_subtree(self.sons, -1)

    def compute_each_layers(self):
        """广度优先遍历计算每一层的节点和序号"""
        q = queue.Queue()
        q.put(self.con_Tree)
        layers = []
        fathers_list = []
        sons_list = []
        while not q.empty():
            n = q.get()
            if n.name in sons_list:
                layers.append(fathers_list)
                fathers_list = sons_list
                sons_list = []
            elif n.name in fathers_list:
                pass
            else:
                fathers_list.append(n.name)
                #print(1)
            for son in n.sons:
                sons_list.append(son.name)
                q.put(son)
        layers.append(fathers_list)
        layers[0] = self.Ux_node
        return layers


if __name__ == '__main__':
    #           0  1  2  3  4  5  6  7  8  9  10 11
    g = Graph([[0, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0],  # 0
               [0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0],  # 1
               [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],  # 2
               [0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0],  # 3
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],  # 4
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 5
               [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],  # 6
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],  # 7
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],  # 10
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # 11
    # g.visualize()
    t = Tree([0, 1], g)
    print(t.compute_each_layers())
