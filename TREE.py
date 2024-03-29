import networkx as nx
import graphviz


class TREE():

    def __init__(self, h):
        '''
        :param h: height of binary decision tree
        Binary Height h Tree Information including P_v and CHILD_v for each vertex
        Also pass initial information for plotting assigned tree
        '''
        # Initialize tree for modeltype
        self.height = h
        self.G = nx.generators.balanced_tree(r=2, h=h)
        self.DG = nx.DiGraph(self.G)  # bi-directed version of G
        hidden_edges = [(i, j) for (i, j) in self.DG.edges if i > j]
        self.DG_prime = nx.restricted_view(self.DG, [], hidden_edges)

        # Set of branching and leaf nodes
        self.L = []
        self.B = []
        for node in self.G.nodes:
            if nx.degree(self.G, node) == 1:
                self.L.append(node)
            elif nx.degree(self.G, node) > 1:
                self.B.append(node)
            else:
                print("Error: tree must not have isolated vertices!")
        self.V = self.B + self.L

        # Tree node relation information
        self.path = nx.single_source_shortest_path(self.DG_prime, 0)
        self.child = {n: list(nx.descendants(self.DG_prime, n)) for n in self.DG_prime.nodes}
        self.direct_ancestor = {n: self.path[n][-2] for n in self.DG_prime.nodes if n != 0}
        self.direct_ancestor[0] = 0
        self.successor = {n: list(self.DG_prime.successors(n)) for n in self.DG_prime.nodes}
        self.LC = {n: v for v in self.DG_prime.nodes
                   for n in self.B
                   if (v % 2 == 1 and v in self.successor[n])}
        self.RC = {n: v for v in self.DG_prime.nodes
                   for n in self.B
                   if (v % 2 == 0 and v in self.successor[n])}
        self.depth = nx.shortest_path_length(self.DG_prime, 0)
        self.levels = list(range(h + 1))
        self.node_level = {level: [n for n in self.DG_prime.nodes if self.depth[n] == level] for level in self.levels}

        # Empty objects for node assignment information to be updated after solving modeltype
        self.color_map = []
        self.labels = {n: None for n in self.DG_prime.nodes}
        # self.pos = nx.nx_pydot.graphviz_layout(self.DG_prime, prog="dot", root=0)

        self.branch_nodes = {}
        self.class_nodes = {}
        self.pruned_nodes = {}
        self.a_v = {}
        self.c_v = {}
