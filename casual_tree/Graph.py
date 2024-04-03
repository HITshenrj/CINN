import networkx as nx
import matplotlib.pyplot as plt
import math
import queue


class LoopException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        print("Graph has Loop!")


class Edge(object):
    def __init__(self, head, tail, weight):
        """
        :param head: int
        :param tail: int
        :param weight: float
        """
        self.head = head
        self.tail = tail
        self.weight = weight

    def __eq__(self, other):
        if self.head == other.head and self.tail == other.tail:
            return True
        else:
            return False


class Node(object):
    def __init__(self, name):
        self.name = name
        self.edges = []     #edge 只有出去的边
        self.tails = []     #tails 指向的节点集
        self.heads = []

    def __eq__(self, other):
        if other.name == self.name:
            return True
        else:
            return False

    def add_edge(self, n, weight):
        e = Edge(self.name, n.name, weight)
        self.edges.append(e)
        self.tails.append(n)
        return e

    def check_pairs(self):
        for i, e in enumerate(self.edges):
            if e.tail != self.tails[i]:
                return False
        return True


class Graph(object):
    def __init__(self, adjacency_matrix=None):
        """
        :param adjacency_matrix: 邻接矩阵
        """
        self.Vs = {}
        self.Es = {}
        self.adjacency_matrix = adjacency_matrix
        if self.adjacency_matrix is not None and not self.check_loop(self.adjacency_matrix):
            self._construct_graph()
            #self.visualize()
        else:
            raise LoopException()

    def _construct_graph(self):
        """遍历邻接矩阵构建图"""
        for i in range(len(self.adjacency_matrix)):
            self.Vs[i] = Node(i)    #初始化所有Node
        for idx1, line in enumerate(self.adjacency_matrix):
            for idx2, i in enumerate(line):
                if i != 0:
                    e = self.Vs[idx1].add_edge(self.Vs[idx2], i)  # math.log(i))
                    try:
                        self.Vs[idx2].heads.append(self.Vs[idx1])   #加入父亲节点
                    except KeyError:
                        self.Vs[idx2].heads = [self.Vs[idx1]]
                    try:
                        self.Es[idx1].append(e)
                    except KeyError:
                        self.Es[idx1] = [e]


    def get_node_num(self):
        return len(self.Vs)

    def get_edge(self, h, t):
        try:
            for e in self.Es[h]:
                if e.tail == t:
                    return e
        except KeyError:
            return None
        return None

    def get_connected(self, h):
        """
        广度优先查找连通
        :return: 连同的所有节点:list
        """
        visited = []
        q = queue.Queue()
        q.put(h)
        while not q.empty():
            n = q.get()
            visited.append(n)
            try:
                edges = self.Es[n]
            except KeyError:
                continue
            for e in edges:
                if e.tail not in visited:
                    q.put(e.tail)
        return visited

    def visualize(self):
        """可视化"""
        G = nx.Graph()
        # H = nx.path_graph(len(self.Vs))
        # G.add_nodes_from(H)
        edge_list = []
        for node, edges in self.Es.items():
            for e in edges:
                edge_list.append((node + 1, e.tail+1))
        G.add_edges_from(edge_list)
        nx.draw(G, with_labels=True, edge_color='b', node_color='g')
        plt.show()

    @staticmethod
    def check_loop(adj_matrix):
        """检查(非)连通图是否有环,深度优先"""

        def DFS(matrix, visited_param, n):
            """从n节点深度优先遍历"""
            for idx, i in enumerate(matrix[n]):
                if i != 0:
                    if visited_param[idx] == 1:
                        return True
                    elif visited_param[idx] == 0:
                        visited_param[idx] = 1
                        if DFS(matrix, visited_param, idx):
                            return True
                    else:
                        pass
            visited_param[n] = 2
            return False

        for i, line in enumerate(adj_matrix):
            visited = [0 for i in range(len(adj_matrix))]  # 0未被访问, 1正在被访问后代节点, 2已经访问完毕
            if DFS(adj_matrix, visited, i):
                return True

        return False


if __name__ == '__main__':
    g = Graph([[0, 2, 1, 3, 0, 0],
               [0, 0, 0, 0, 2, 0],
               [0, 0, 0, 0, 0, 8],
               [0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0, 0]])
    print(g.get_connected(0))