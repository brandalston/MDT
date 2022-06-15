import numpy as np
import pandas as pd
import random
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from gurobipy import *
import networkx as nx
import csv


def get_data(file_name, target):
    # Return dataset from 'file_name' in Pandas dataframe
    # dataset located in workspace folder named 'Datasets'
    # Ensure all features are in [0,1] through encoding process
    global data_processed, data
    # try:
    cols_dict = {
        'auto-mpg': ['target', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year',
                     'origin', 'car_name'],
        'balance-scale': ['target', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
        'banknote_authentication': ['variance-of-wavelet', 'skewness-of-wavelet', 'curtosis-of-wavelet', 'entropy',
                                    'target'],
        'blood_transfusion': ['R', 'F', 'M', 'T', 'target'],
        'breast-cancer': ['target', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig',
                          'breast', 'breast-quad', 'irradiat'],
        'car': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'],
        'climate': ['Study', 'Run', 'vconst_corr', 'vconst_2', 'vconst_3', 'vconst_4', 'vconst_5', 'vconst_7',
                    'ah_corr', 'ah_bolus', 'slm_corr', 'efficiency_factor', 'tidal_mix_max', 'vertical_decay_scale',
                    'convect_corr', 'bckgrnd_vdc1', 'bckgrnd_vdc_ban', 'bckgrnd_vdc_eq', 'bckgrnd_vdc_psim',
                    'Prandtl', 'target'],
        'flare1': ['class', 'largest-spot-size', 'spot-distribution', 'activity', 'evolution',
                   'previous-24hr-activity', 'historically-complex', 'become-h-c', 'area', 'area-largest-spot',
                   'c-target', 'm-target', 'x-target'],
        'flare2': ['class', 'largest-spot-size', 'spot-distribution', 'activity', 'evolution',
                   'previous-24hr-activity', 'historically-complex', 'become-h-c', 'area', 'area-largest-spot',
                   'c-target', 'm-target', 'x-target'],
        'glass': ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'target'],
        'hayes-roth': ['file_name', 'hobby', 'age', 'educational-level', 'marital-status', 'target'],
        'house-votes-84': ['target', 'handicapped-infants', 'water-project-cost-sharing',
                           'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                           'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                           'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
                           'superfund-right-to-sue', 'crime', 'duty-free-exports',
                           'export-administration-act-south-africa'],
        'image_segmentation': ['target', 'region-centroid-col', 'region-centroid-row', 'region-pixel-count',
                               'short-line-density-5', 'short-line-density-2', 'vedge-mean', 'vegde-sd',
                               'hedge-mean', 'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean',
                               'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean',
                               'saturatoin-mean', 'hue-mean'],
        'ionosphere': list(range(1, 35)) + ['target'],
        'iris': ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target'],
        'kr-vs-kp': ['bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp', 'bxqsq',
                     'cntxt', 'dsopp', 'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8', 'reskd', 'reskr',
                     'rimmx', 'rkxwp', 'rxmsq', 'simpl', 'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk',
                     'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg', 'target'],
        'monk1': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'monk2': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'monk3': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'parkinsons': ['file_name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                       'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'target', 'RPDE',
                       'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
        'soybean-small': list(range(1, 36)) + ['target'],
        'tic-tac-toe': ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square',
                        'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square',
                        'bottom-right-square', 'target'],
        'wine-red': ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
                     'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                     'target'],
        'wine-white': ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
                       'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                       'target']
    }
    numerical_datasets = ['iris', 'wine-red', 'wine-white', 'banknote_authentication', 'blood_transfusion',
                          'climate', 'glass', 'image_segmentation', 'ionosphere', 'parkinsons']
    categorical_datasets = ['balance-scale', 'car', 'kr-vs-kp', 'house-votes-84', 'tic-tac-toe',
                            'hayes-roth', 'monk1', 'monk2', 'monk3', 'soybean-small', 'breast-cancer']
    if file_name in cols_dict:
        if file_name not in ['glass', 'hayes-roth', 'parkinsons']:
            data = pd.read_csv('Datasets/' + file_name + '.data', names=cols_dict[file_name])
        elif file_name == 'hayes-roth' or 'parkinsons':
            data = pd.read_csv('Datasets/' + file_name + '.data', names=cols_dict[file_name], index_col='file_name')
        elif file_name == 'glass':
            data = pd.read_csv('Datasets/' + file_name + '.data', names=cols_dict[file_name], index_col='Id')
    if file_name in numerical_datasets:
        data_processed = preprocess(data, numerical_features=data.columns != target)
    if file_name in categorical_datasets:
        data_processed = preprocess(data, categorical_features=data.columns != target)
    data_processed.name = file_name
    data_processed['target'] = data['target']
    return data_processed
    # except:
        # print("Dataset not found or error in preprocess!\n")
        # return


class CandidateThresholdBinarizer(TransformerMixin, BaseEstimator):
    """ Binarize continuous data using candidate thresholds.

    For each feature, sort observations by values of that feature, then find
    pairs of consecutive observations that have different class labels and
    different feature values, and define a candidate threshold as the average of
    these two observations’ feature values.

    Attributes
    ----------
    candidate_thresholds : dict mapping features to list of thresholds
    """

    def __init__(self):
        self.candidate_thresholds = {}

    def fit(self, data, target):
        """ Finds all candidate split thresholds for each feature.

        Parameters
        ----------
        data : pandas DataFrame with observations and labels
        target: column name of labels

        Returns
        -------
        self
        """
        for f in data.columns:
            if f == target: continue
            thresholds = []
            # Sort by feature value, then by label
            sorted_data = data.sort_values([f, target])
            prev_feature_val, prev_label = sorted_data.iloc[0][f], sorted_data.iloc[0][target]
            for idx, row in sorted_data.iterrows():
                curr_feature_val, curr_label = row[f], row[target]
                if (curr_label != prev_label and
                        not math.isclose(curr_feature_val, prev_feature_val)):
                    thresh = (prev_feature_val + curr_feature_val) / 2
                    thresholds.append(thresh)
                prev_feature_val, prev_label = curr_feature_val, curr_label
            self.candidate_thresholds[f] = thresholds
        return self

    def transform(self, data, target):
        """
        Binarize numerical features using candidate thresholds.

        Parameters
        ----------
        data : pandas DataFrame with observations
        target: column name of data labels

        Returns
        -------
        data_binarized : pandas DataFrame that is the result of binarizing X
        """
        check_is_fitted(self)
        data_binarized = pd.DataFrame()
        for f in data.columns:
            if f == target: continue
            for threshold in self.candidate_thresholds[f]:
                data_binarized[f'{f} <= {threshold}'] = (data[f] <= threshold)
        return data_binarized


def preprocess(data, numerical_features=None, categorical_features=None, binarization=None):
    if numerical_features is None:
        numerical_features = []
    if categorical_features is None:
        categorical_features = []
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    numerical_transformer = MinMaxScaler()
    if binarization == 'binning':
        numerical_transformer = KBinsDiscretizer(encode='onehot-dense')
    elif binarization == 'candidate':
        numerical_transformer = CandidateThresholdBinarizer()
    ct = ColumnTransformer([("num", numerical_transformer, numerical_features),
                            ("cat", categorical_transformer, categorical_features)])
    data_new = pd.DataFrame(ct.fit_transform(data), index=data.index, columns=ct.get_feature_names_out())
    return data_new


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
        global cc_L, cc_R
        if not np.array_equal(np.unique(data.svm), [-1, 1]):
            print("Class labels must be -1 and +1")
            raise ValueError
        feature_set = [f for f in data.columns if f != 'svm']
        # Define left and right index sets according to SVM class
        Lv_I = set(i for i in data.index if data.at[i, 'svm'] == -1)
        Rv_I = set(i for i in data.index if data.at[i, 'svm'] == +1)
        # Remove any points in each index set whose feature_set are equivalent to some
        common_points_L, common_points_R = set(), set()
        for x in Lv_I:
            for y in Rv_I:
                if data.loc[x, feature_set].equals(data.loc[y, feature_set]):
                    common_points_L.add(x)
                    common_points_R.add(y)
        Lv_I -= common_points_L
        Rv_I -= common_points_R
        data = data.drop(common_points_L|common_points_R)

        # Find separating hyperplane by solving dual of Lagrangian of the standard hard margin linear SVM problem
        try:
            m = Model("HM_Linear_SVM")
            m.Params.LogToConsole = 0
            m.Params.NumericFocus = 3
            alpha = m.addVars(data.index, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
            W = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

            m.addConstrs(W[f] == quicksum(alpha[i] * data.at[i, 'svm'] * data.at[i, f] for i in data.index)
                         for f in feature_set)
            m.addConstr(quicksum(alpha[i] * data.at[i, 'svm'] for i in data.index) == 0)
            m.setObjective(alpha.sum() - (1 / 2) * quicksum(W[f] * W[f] for f in feature_set), GRB.MAXIMIZE)
            m.optimize()

            # Any i with positive alpha[i] works
            for i in data.index:
                if alpha[i].x > m.Params.FeasibilityTol:
                    b = data.at[i, 'svm'] - sum(W[f].x * data.at[i, f] for f in feature_set)
                    break
            a_v = {f: W[f].x for f in feature_set}
            c_v = -b  # Must flip intercept because of how QP was setup
            self.a_v, self.c_v = a_v, c_v
            return self
        except Exception:
            # Find separating hyperplane by solving dual of Lagrangian of standard soft margin linear SVM problem
            try:
                # Find any points in Lv_I, Rv_I which are convex combinations of the other set
                cc_L, cc_R = set(), set()
                for i in Lv_I:
                    convex_combo = Model("Left Index Convex Combination")
                    convex_combo.Params.LogToConsole = 0
                    lambdas = convex_combo.addVars(Rv_I, vtype=GRB.CONTINUOUS, lb=0)
                    convex_combo.addConstrs(quicksum(lambdas[i]*data.at[i, f] for i in Rv_I) == data.at[i, f]
                                            for f in feature_set)
                    convex_combo.addConstr(lambdas.sum() == 1)
                    convex_combo.setObjective(0, GRB.MINIMIZE)
                    convex_combo.optimize()
                    if convex_combo.Status != GRB.INFEASIBLE:
                        cc_L.add(i)
                for i in Rv_I:
                    convex_combo = Model("Right Index Convex Combination")
                    convex_combo.Params.LogToConsole = 0
                    lambdas = convex_combo.addVars(Lv_I, vtype=GRB.CONTINUOUS, lb=0)
                    convex_combo.addConstrs(quicksum(lambdas[i]*data.at[i, f] for i in Lv_I) == data.at[i, f]
                                            for f in feature_set)
                    convex_combo.addConstr(lambdas.sum() == 1)
                    convex_combo.setObjective(0, GRB.MINIMIZE)
                    convex_combo.optimize()
                    if convex_combo.Status != GRB.INFEASIBLE:
                        cc_R.add(i)

                # Find noramlized max inner product of convex combinations
                # to use as upper bound in dual of Lagrangian of soft margin SVM
                margin_ub = GRB.INFINITY
                inner_products = {item: np.inner(data.loc[item[0], feature_set],
                                          data.loc[item[1], feature_set])
                                  for item in list(combinations(cc_L|cc_R, 2))}
                if inner_products:
                    margin_ub = max(inner_products.values()) / \
                                min(len(cc_L | cc_R), np.linalg.norm(list(inner_products.values()), 2))

                # Solve dual of Lagrangian of soft margin SVM
                m = Model("SM_Linear_SVM")
                m.Params.LogToConsole = 0
                m.Params.NumericFocus = 3
                alpha = m.addVars(data.index, vtype=GRB.CONTINUOUS, lb=0, ub=margin_ub)
                W = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

                m.addConstrs(W[f] == quicksum(alpha[i] * data.at[i, 'svm'] * data.at[i, f] for i in data.index)
                             for f in feature_set)
                m.addConstr(quicksum(alpha[i] * data.at[i, 'svm'] for i in data.index) == 0)
                m.setObjective(alpha.sum() - (1 / 2) * quicksum(W[f] * W[f] for f in feature_set), GRB.MAXIMIZE)
                m.optimize()

                # Any i with positive alpha[i] works
                for i in data.index:
                    if alpha[i].x > m.Params.FeasibilityTol:
                        b = data.at[i, 'svm'] - sum(W[f].x * data.at[i, f] for f in feature_set)
                        break
                a_v = {f: W[f].x for f in feature_set}
                c_v = -b  # Must flip intercept because of how QP was setup
                self.a_v, self.c_v = a_v, c_v
                return self
            except Exception:
                # Find any generic hard margin separating hyperplane
                try:
                    gen_hyperplane = Model("Separating hyperplane")
                    gen_hyperplane.Params.LogToConsole = 0
                    a_hyperplane = gen_hyperplane.addVars(feature_set, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    c_hyperplane = gen_hyperplane.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    gen_hyperplane.addConstrs(
                        quicksum(a_hyperplane[f] * data.at[i, f] for f in feature_set) + 1 <= c_hyperplane
                        for i in Lv_I)
                    gen_hyperplane.addConstrs(
                        quicksum(a_hyperplane[f] * data.at[i, f] for f in feature_set) - 1 >= c_hyperplane
                        for i in Rv_I)
                    gen_hyperplane.setObjective(0, GRB.MINIMIZE)
                    gen_hyperplane.optimize()

                    a_v = {f: a_hyperplane[f].X for f in feature_set}
                    c_v = c_hyperplane.X
                    self.a_v, self.c_v = a_v, c_v
                    return self
                except Exception:
                    # Find any generic separating hyperplane
                    try:
                        gen_hyperplane = Model("Separating hyperplane")
                        gen_hyperplane.Params.LogToConsole = 0
                        a_hyperplane = gen_hyperplane.addVars(feature_set, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                        c_hyperplane = gen_hyperplane.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                        gen_hyperplane.addConstrs(
                            quicksum(a_hyperplane[f] * data.at[i, f] for f in feature_set) <= c_hyperplane
                            for i in Lv_I)
                        gen_hyperplane.addConstrs(
                            quicksum(a_hyperplane[f] * data.at[i, f] for f in feature_set) >= c_hyperplane
                            for i in Rv_I)
                        gen_hyperplane.setObjective(0, GRB.MINIMIZE)
                        gen_hyperplane.optimize()

                        if gen_hyperplane.status != GRB.INFEASIBLE:
                            a_v = {f: a_hyperplane[f].X for f in feature_set}
                            c_v = c_hyperplane.X
                            self.a_v, self.c_v = a_v, c_v
                        return self
                    except Exception:
                        # Return random separating hyperplane
                        a_v = {f: random.random() for f in feature_set}
                        c_v = random.random()
                        self.a_v, self.c_v = a_v, c_v
                        return self


def model_results(model, tree):
    # Print assigned branching, classification, and pruned nodes of tree

    for v in tree.V:
        if model._P[v].x > 0.5:
            for k in model._data[model._target].unique():
                if model._W[v, k].x > 0.5:
                    print('Vertex ' + str(v) + ' class ' + str(k))
        elif model._P[v].x < 0.5 and model._B[v].x > 0.5:
            print('Vertex ' + str(v) + ' branching', tree.DG_prime.nodes[v]['branching'])
        elif model._P[v].x < 0.5 and model._B[v].x < 0.5:
            print('Vertex ' + str(v) + ' pruned')

    # Print datapoint paths through tree
    for i in sorted(model._data.index):
        path = [0]
        for v in model._tree.V:
            if model._Q[i, v].x > 0.5:
                path.append(v)
                if model._S[i, v].x > 0.5:
                    print('Datapoint ' + str(i) + ' correctly assigned class ' + str(model._data.at[i, model._target])
                         + ' at ' + str(v) + '. Path: ', path)
                for k in model._data[model._target].unique():
                    if model._W[v, k].x > 0.5:
                        print('datapoint ' + str(i) + ' incorrectly assigned class ' + str(k)
                              + ' at ' + str(v) + '. Path: ', path)



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
                v = tree.LC[v] if sum(a_v[f] * data.at[i, f] for f in data.columns if f != target) <= c_v else tree.RC[
                    v]
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
             test_acc/len(test_set), train_acc/len(opt_model.datapoints), opt_model.model.Runtime, opt_model.model.MIPGap,
             opt_model.model.ObjVal, opt_model.model.ObjBound, opt_model.modeltype, opt_model.HP_time,
             opt_model.model._septime, opt_model.model._sepnum, opt_model.model._sepcuts, opt_model.model._sepavg,
             opt_model.model._vistime, opt_model.model._visnum, opt_model.model._viscuts,
             opt_model.eps, opt_model.time_limit, rand_state,
             opt_model.warmstart['use'], opt_model.regularization, opt_model.max_features])
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
    feature_set = [col for col in data.columns if col != target]

    # Top-down random tree
    tree.a_v[0], tree.c_v[0] = {f: random.random() for f in feature_set}, random.random()
    tree.DG_prime.nodes[0]['branching'] = (tree.a_v[0], tree.c_v[0])

    for level in tree.node_level:
        if level == 0: continue
        for v in tree.node_level[level]:
            if 'branching' in tree.DG_prime.nodes[tree.direct_ancestor[v]]:
                if random.random() > .5 and level != tree.height:
                    tree.a_v[v], tree.c_v[v] = {f: random.random() for f in feature_set}, random.random()
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
    feature_set = [col for col in data.columns if col != target]

    # Bottoms-up random tree
    node_list = tree.V.copy()
    tree.a_v[0], tree.c_v[0] = {f: random.random() for f in feature_set}, random.random()
    tree.DG_prime.nodes[0]['branching'] = (tree.a_v[0], tree.c_v[0])
    node_list.remove(0)

    while len(node_list) > 0:
        selected = random.choice(node_list)
        tree.DG_prime.nodes[selected]['class'] = random.choice(classes)
        for v in reversed(tree.path[selected][1:-1]):
            if v in node_list:
                tree.a_v[v], tree.c_v[v] = {f: random.random() for f in feature_set}, random.random()
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
