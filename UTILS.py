import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import networkx as nx
from gurobipy import *


class HM_Linear_SVM():
    """ Hard-margin linear SVM trained using quadratic programming.

    Assumes class labels are -1 and +1, and finds a hyperplane (a, c) such that a'x^i <= c iff y^i = -1.
    If QP fails for whatever reason, just return any separating hyperplane

    Solve dual (generated using Lagrange multipliers) of traditional hard-margin linear SVM
    """

    def SVM_fit(self, data):
        featureset = [col for col in data.columns if col != 'y']
        if not np.array_equal(np.unique(data.y), [-1, 1]):
            print("Class labels must be -1 and +1")
            raise ValueError
        try:
            m = Model("HM_Linear_SVM")
            m.Params.LogToConsole = 0
            m.Params.NumericFocus = 3
            alpha = m.addVars(data.index, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
            W = m.addVars(featureset, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

            m.addConstrs(W[f] == quicksum(alpha[i] * data.at[i, 'y'] * data.at[i, f] for i in data.index)
                         for f in featureset)
            m.addConstr(quicksum(alpha[i] * data.at[i, 'y'] for i in data.index) == 0)
            m.setObjective(alpha.sum() - (1 / 2) * quicksum(W[f] * W[f] for f in featureset), GRB.MAXIMIZE)
            m.optimize()

            # Any i with positive alpha[i] works
            for i in data.index:
                if alpha[i].X > m.Params.FeasibilityTol:
                    b = data.at[i, 'y'] - sum(W[f].X * data.at[i, f] for f in featureset)
                    break
            a_v = {f: W[f].X for f in featureset}
            c_v = -b  # Must flip intercept because of how QP was setup
            self.a_v, self.c_v = a_v, c_v
            return self

        except:
            # If QP fails to solve, return any separating hyperplane
            Lv_I = [i for i in data.index if data.at[i, 'y'] == -1]
            Rv_I = [i for i in data.index if data.at[i, 'y'] == +1]

            m = Model("separating hyperplane")
            m.Params.LogToConsole = 0
            a_hyperplane = m.addVars(featureset, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            c_hyperplane = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.addConstrs(quicksum(a_hyperplane[f] * data.at[i, f] for f in featureset) + 1 <= c_hyperplane for i in Lv_I)
            m.addConstrs(quicksum(a_hyperplane[f] * data.at[i, f] for f in featureset) - 1 >= c_hyperplane for i in Rv_I)
            m.setObjective(0, GRB.MINIMIZE)
            m.optimize()

            a_v = {f: a_hyperplane[f].X for f in featureset}
            c_v = c_hyperplane.X
            self.a_v, self.c_v = a_v, c_v

            return self


def tree_check(tree):
    class_nodes = {v: tree.DG_prime.nodes[v]['class']
                   for v in tree.DG_prime.nodes if 'class' in tree.DG_prime.nodes[v]}
    branch_nodes = {v: tree.DG_prime.nodes[v]['branching']
                    for v in tree.DG_prime.nodes if 'branching' in tree.DG_prime.nodes[v]}
    pruned_nodes = {v: tree.DG_prime.nodes[v]['pruned']
                    for v in tree.DG_prime.nodes if 'pruned' in tree.DG_prime.nodes[v]}
    for v in class_nodes.keys():
        if not (all(n in branch_nodes.keys() for n in tree.path[v][:-1])):
            return False
        if not (all(c in pruned_nodes.keys() for c in tree.child[v])):
            return False


def data_predict(tree, data, target):
    # get branching and class node and direct children of each node
    branching_nodes = nx.get_node_attributes(tree.DG_prime, 'branch on feature')
    class_nodes = nx.get_node_attributes(tree.DG_prime, 'class')
    acc = 0
    results = {i: [None, []] for i in data.index}

    for i in data.index:
        v = 0
        while results[i][0] is None:
            results[i][1].append(v)
            if v in branching_nodes:
                (a_v, c_v) = branching_nodes[v]
                if sum(a_v[f]*data.at[i, f] for f in data.columns if f != target) <= c_v:
                    v = tree.LC[v]
                else: v = tree.RC[v]
            elif v in class_nodes:
                results[i][0] = class_nodes[v]
                if results[i][0] == data.at[i, target]:
                    acc += 1
                    results[i].append('correct')
                else:
                    results[i].append('incorrect')
            else:
                results[i][0] = 'ERROR'

    return

