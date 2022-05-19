import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from gurobipy import *
import networkx as nx
import csv


def get_data(name, target):
    # Return dataset from 'name' in Pandas dataframe
    # dataset located in workspace folder named 'Datasets'
    # Ensure all features are in [0,1] through encoding process
    global data
    try:
        data = pd.read_csv('Datasets/' + name + '.csv', na_values='?')
        data = preprocess(data, target)
    except:
        print("Dataset Not Found or Error in Encoding Process!")
    return data


def preprocess(data, target):
    new_data = pd.DataFrame(index=data.index, columns=data.columns)
    numeric_col = [f for f in data.columns if target != f and np.issubdtype(data[f].dtype, np.number)]
    cat_col = [f for f in data.columns if (f in data.select_dtypes(include='object').columns or
                                           f in data.select_dtypes(include='category').columns)
               and f != target]
    for f in numeric_col:
        if 0 <= data[f].min() and data[f].max() <= 1:
            new_data[f] = data[f]
        elif len(data[f].unique()) == 1:
            new_data[f] = LabelEncoder().fit_transform(data[f])
        else:
            new_data[f] = (data[f] - data[f].min()) / (data[f].max() - data[f].min())
    for f in cat_col:
        new_data[f] = LabelEncoder().fit_transform(data[f])
        new_data[f] = (new_data[f] - new_data[f].min()) / (new_data[f].max() - new_data[f].min())

    new_data[target] = data[target]

    return new_data


class Linear_Separator():
    """ Hard-margin linear SVM trained using quadratic programming.

    Assumes class labels are -1 and +1, and finds a hyperplane (a, c) such that a'x^i <= c iff y^i = -1.
    If QP fails for whatever reason, just return any separating hyperplane

    Solve dual (generated using Lagrange multipliers) of traditional hard-margin linear SVM
    """

    def __init__(self):
        self.a_v = 0
        self.c_v = 0

    def SVM_fit(self, data):
        print('Finding (a,c)')
        featureset = [col for col in data.columns if col != 'svm']
        if not np.array_equal(np.unique(data.svm), [-1, 1]):
            print("Class labels must be -1 and +1")
            raise ValueError
        try:
            m = Model("HM_Linear_SVM")
            m.Params.LogToConsole = 0
            m.Params.NumericFocus = 3
            alpha = m.addVars(data.index, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
            W = m.addVars(featureset, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

            m.addConstrs(W[f] == quicksum(alpha[i] * data.at[i, 'svm'] * data.at[i, f] for i in data.index)
                         for f in featureset)
            m.addConstr(quicksum(alpha[i] * data.at[i, 'svm'] for i in data.index) == 0)
            m.setObjective(alpha.sum() - (1 / 2) * quicksum(W[f] * W[f] for f in featureset), GRB.MAXIMIZE)
            m.optimize()

            # Any i with positive alpha[i] works
            for i in data.index:
                if alpha[i].x > m.Params.FeasibilityTol:
                    b = data.at[i, 'svm'] - sum(W[f].x * data.at[i, f] for f in featureset)
                    break
            a_v = {f: W[f].x for f in featureset}
            c_v = -b  # Must flip intercept because of how QP was setup
            print('Solved SVM problem')
            self.a_v, self.c_v = a_v, c_v
            return self
        except Exception:
            print('Generating any separating hyperplane')
            # If QP fails to solve, return any separating hyperplane
            Lv_I = [i for i in data.index if data.at[i, 'svm'] == -1]
            Rv_I = [i for i in data.index if data.at[i, 'svm'] == +1]
            print(Lv_I)
            print(Rv_I)
            m_hyperplane = Model("Separating hyperplane")
            m_hyperplane.Params.LogToConsole = 1
            a_hyperplane = m_hyperplane.addVars(featureset, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            c_hyperplane = m_hyperplane.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

            m_hyperplane.addConstrs(
                quicksum(a_hyperplane[f] * data.at[i, f] for f in featureset) + 1 <= c_hyperplane for i in Lv_I)
            m_hyperplane.addConstrs(
                quicksum(a_hyperplane[f] * data.at[i, f] for f in featureset) - 1 >= c_hyperplane for i in Rv_I)
            m_hyperplane.setObjective(0, GRB.MINIMIZE)
            m_hyperplane.optimize()

            if m_hyperplane.status == GRB.OPTIMAL:
                a_v = {f: a_hyperplane[f].x for f in featureset}
                c_v = c_hyperplane.X
                self.a_v, self.c_v = a_v, c_v
                print('found a generic separating hyperplane')
            return self


def model_results(model):
    # Print assigned branching, classification, and pruned nodes of tree
    for v in model._tree.V:
        if model._P[v].x > 0.5:
            for k in model._data[model._target].unique():
                if model._W[v, k].x > 0.5:
                    print('Vertex ' + str(v) + ' class ' + str(k))
        elif model._P[v].x < 0.5 and model._B[v].x > 0.5:
            print('Vertex ' + str(v) + ' branching', model._tree.DG_prime.nodes[v]['branching'])
        elif model._P[v].x < 0.5 and model._B[v].x < 0.5:
            print('Vertex ' + str(v) + 'pruned')

    # Print datapoint paths through tree
    for i in model._data.index:
        path = []
        for v in model._tree.V:
            if model._Q[i, v].x > 0.5:
                path.append(v)
                if model._S[i, v].x > 0.5:
                    print('datapoint ' + str(i) + ' correctly assigned class ' + str(model._data.at[i, model._target])
                          + ' at ' + str(v) + '. Path: ', path)
                '''
                elif model._S[i, v].x < 0.5:
                    for k in model._data[model._target].unique():
                        if model._W[v, k].x > 0.5:
                            print('datapoint ' + str(i) + ' incorrectly assigned class ' + str(k)
                                  + ' at ' + str(v) + '. Path: ', path)
                '''


def tree_check(tree):
    class_nodes = {v: tree.DG_prime.nodes[v]['class']
                   for v in tree.DG_prime.nodes if 'class' in tree.DG_prime.nodes[v]}
    branch_nodes = {v: tree.DG_prime.nodes[v]['branching']
                    for v in tree.DG_prime.nodes if 'branching' in tree.DG_prime.nodes[v]}
    pruned_nodes = {v: tree.DG_prime.nodes[v]['pruned']
                    for v in tree.DG_prime.nodes if 'pruned' in tree.DG_prime.nodes[v]}
    for v in class_nodes:
        if not (all(n in branch_nodes for n in tree.path[v][:-1])):
            return False
        if not (all(c in pruned_nodes for c in tree.child[v])):
            return False
    return True


def data_predict(tree, data, target):
    # Ensure assigned tree is valid
    if not tree_check(tree): print('Tree is invalid')
    # Get branching and class node assignments of tree
    branching_nodes = nx.get_node_attributes(tree.DG_prime, 'branching')
    class_nodes = nx.get_node_attributes(tree.DG_prime, 'class')
    # Results dictionary for datapoint assignments and path through tree
    acc = 0
    results = {i: [None, []] for i in data.index}

    for i in data.index:
        v = 0
        while results[i][0] is None:
            results[i][1].append(v)
            if v in branching_nodes:
                (a_v, c_v) = tree.DG_prime.nodes[v]['branching']
                v = tree.LC[v] if sum(a_v[f]*data.at[i, f] for f in data.columns if f != target) <= c_v else tree.RC[v]
            elif v in class_nodes:
                results[i][0] = class_nodes[v]
                if results[i][0] == data.at[i, target]:
                    acc += 1
                    results[i].append('correct')
                else:
                    results[i].append('incorrect')
            else:
                results[i][0] = 'ERROR'

    return acc, results


def model_summary(opt_model, tree, test_set, rand_state, results_file):
    # Ensure Tree Valid
    if not tree_check(tree): print('Invalid Tree!!')
    # Test / Train Acc
    test_acc, test_assignments = data_predict(tree=tree, data=test_set, target=opt_model.target)
    train_acc, train_assignments = data_predict(tree=tree, data=opt_model.data, target=opt_model.target)

    # Update .csv file with modeltype metrics
    with open(results_file, mode='a') as results:
        results_writer = csv.writer(results, delimiter=',', quotechar='"')
        results_writer.writerow(
            [opt_model.dataname, tree.height, len(opt_model.datapoints),
             test_acc / len(test_set), train_acc / len(opt_model.datapoints), opt_model.model.Runtime,
             opt_model.model.MIPGap, opt_model.model.ObjVal, opt_model.model.ObjBound, opt_model.modeltype,
             opt_model.model._septime, opt_model.model._sepnum, opt_model.model._sepcuts, opt_model.model._sepavg,
             opt_model.model._vistime, opt_model.model._visnum, opt_model.model._viscuts,
             opt_model.eps, opt_model.time_limit, rand_state, opt_model.warmstart])
        results.close()


def random_tree(tree, data, target):
    # Clear any existing node assignments
    for v in tree.V:
        if 'class' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['class']
        if 'branching' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['branching']
        if 'pruned' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['pruned']

    TD_best_acc = -1
    BU_best_acc = -1
    for i in range(50):
        TD_tree = TD_rand_tree(tree, data, target)
        TD_acc, TD_results = data_predict(TD_tree, data, target)
        if TD_acc > TD_best_acc:
            TD_best_acc = TD_acc
            TD_best_results = TD_results
            best_TD_tree = TD_tree

    for i in range(50):
        BU_tree = BU_rand_tree(tree, data, target)
        BU_acc, BU_results = data_predict(BU_tree, data, target)
        if BU_acc > BU_best_acc:
            BU_best_acc = BU_acc
            BU_best_results = BU_results
            best_BU_tree = BU_tree
    if TD_best_acc > BU_best_acc:
        return {'tree': best_TD_tree, 'results': TD_best_results}
    else:
        return {'tree': best_BU_tree, 'results': BU_best_results}


def TD_rand_tree(tree, data, target):
    # Clear any existing node assignments
    for v in tree.V:
        if 'class' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['class']
        if 'branching' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['branching']
        if 'pruned' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['pruned']
    classes = data[target].unique()
    featureset = [col for col in data.columns if col != target]

    # Top-down random tree
    tree.a_v[0], tree.c_v[0] = {f: random.random() for f in featureset}, random.random()
    tree.DG_prime.nodes[0]['branching'] = (tree.a_v[0], tree.c_v[0])

    for level in tree.node_level:
        if level == 0: continue
        for v in tree.node_level[level]:
            if 'branching' in tree.DG_prime.nodes[tree.direct_ancestor[v]]:
                if random.random() > .5 and level != tree.height:
                    tree.a_v[v], tree.c_v[v] = {f: random.random() for f in featureset}, random.random()
                    tree.DG_prime.nodes[v]['branching'] = (tree.a_v[v], tree.c_v[v])
                else:
                    tree.DG_prime.nodes[v]['class'] = random.choice(classes)
            else:
                tree.DG_prime.nodes[v]['pruned'] = 0
    return tree


def BU_rand_tree(tree, data, target):
    # Clear any existing node assignments
    for v in tree.V:
        if 'class' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['class']
        if 'branching' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['branching']
        if 'pruned' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['pruned']

    classes = data[target].unique()
    featureset = [col for col in data.columns if col != target]

    # Bottoms-up random tree
    node_list = tree.V.copy()
    tree.a_v[0], tree.c_v[0] = {f: random.random() for f in featureset}, random.random()
    tree.DG_prime.nodes[0]['branching'] = (tree.a_v[0], tree.c_v[0])
    node_list.remove(0)

    while len(node_list) > 0:
        selected = random.choice(node_list)
        tree.DG_prime.nodes[selected]['class'] = random.choice(classes)
        for v in reversed(tree.path[selected][1:-1]):
            if v in node_list:
                tree.a_v[v], tree.c_v[v] = {f: random.random() for f in featureset}, random.random()
                tree.DG_prime.nodes[v]['branching'] = (tree.a_v[v], tree.c_v[v])
                node_list.remove(v)
            else:
                break
        for c in tree.child[selected]:
            if c in node_list:
                tree.DG_prime.nodes[c]['pruned'] = 0
                node_list.remove(c)
            else:
                break
        node_list.remove(selected)

    return tree


class consol_log:
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Handles the flush command by doing nothing and needed for python3 compatability
        # Specify extra behavior here
        pass
