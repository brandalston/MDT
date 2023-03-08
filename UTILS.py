import csv, random
import numpy as np
from gurobipy import *
from data_load import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted


def get_data(dataset, binarization=None):
    datasetloadfcn = {
        'balance_scale': load_balance_scale,
        'banknote': load_banknote_authentication,
        'blood': load_blood_transfusion,
        'breast_cancer': load_breast_cancer,
        'car': load_car_evaluation,
        'kr_vs_kp': load_chess,
        'climate': load_climate_model_crashes,
        'house_votes_84': load_house_votes_84,
        'fico_binary': load_fico_binary,
        'glass': load_glass_identification,
        'hayes_roth': load_hayes_roth,
        'image': load_image_segmentation,
        'ionosphere': load_ionosphere,
        'iris': load_iris,
        'monk1': load_monk1,
        'monk2': load_monk2,
        'monk3': load_monk3,
        'parkinsons': load_parkinsons,
        'soybean_small': load_soybean_small,
        'spect': load_spect,
        'tic_tac_toe': load_tictactoe_endgame,
        'wine_red': load_wine_red,
        'wine_white': load_wine_white
    }

    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red',
                          'glass', 'image', 'ionosphere', 'parkinsons']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    already_processed = ['fico_binary']
    load_function = datasetloadfcn[dataset]
    X, y = load_function()
    """ We assume the target column of dataset is labeled 'target'
        Change value at your discretion """
    codes, uniques = pd.factorize(y)
    y = pd.Series(codes, name='target')
    if dataset in categorical_datasets:
        X_new, ct = preprocess(X, categorical_features=X.columns)
        X_new = pd.DataFrame(X_new, columns=ct.get_feature_names_out(X.columns))
        X_new.columns = X_new.columns.str.replace('cat__', '')
    else:
        X_new = X
        if dataset in numerical_datasets:
            if binarization is None:
                X_new, ct = preprocess(X, numerical_features=X.columns)
                X_new = pd.DataFrame(X_new, columns=X.columns)
            else:
                X_new, ct = preprocess(X, y=y, binarization=binarization, numerical_features=X.columns)
                cols = []
                for key in ct.transformers_[0][1].candidate_thresholds_:
                    for item in ct.transformers_[0][1].candidate_thresholds_[key]:
                        cols.append(f"{key}<={item}")
                X_new = pd.DataFrame(X_new, columns=cols)
    if dataset in categorical_datasets: X_new = X_new.astype(int)
    data_new = pd.concat([X_new, y], axis=1)
    return data_new


def preprocess(X, y=None, numerical_features=None, categorical_features=None, binarization=None):
    """ Preprocess a dataset.

    Numerical features are scaled to the [0,1] interval by default, but can also
    be binarized, either by considering all candidate thresholds for a
    univariate split, or by binning. Categorical features are one-hot encoded.

    Parameters
    ----------
    X
    X_test
    y_train : pandas Series of training labels, only needed for binarization
        with candidate thresholds
    numerical_features : list of numerical features
    categorical_features : list of categorical features
    binarization : {'all-candidates', 'binning'}, default=None
        Binarization technique for numerical features.
        all-candidates
            Use all candidate thresholds.
        binning
            Perform binning using scikit-learn's KBinsDiscretizer.
        None
            No binarization is performed, features scaled to the [0,1] interval.

    Returns
    -------
    X_train_new : pandas DataFrame that is the result of binarizing X
    """

    if numerical_features is None:
        numerical_features = []
    if categorical_features is None:
        categorical_features = []

    numerical_transformer = MinMaxScaler()
    if binarization == 'all-candidates':
        numerical_transformer = CandidateThresholdBinarizer()
    elif binarization == 'binning':
        numerical_transformer = KBinsDiscretizer(encode='onehot-dense')
    elif binarization is None:
        numerical_transformer = MinMaxScaler()
    # categorical_transformer = OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='ignore') # Should work in scikit-learn 1.0
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ct = ColumnTransformer([("num", numerical_transformer, numerical_features),
                            ("cat", categorical_transformer, categorical_features)])
    X_train_new = ct.fit_transform(X, y)

    return X_train_new, ct


class CandidateThresholdBinarizer(TransformerMixin, BaseEstimator):
    """ Binarize continuous data using candidate thresholds.

    For each feature, sort observations by values of that feature, then find
    pairs of consecutive observations that have different class labels and
    different feature values, and define a candidate threshold as the average of
    these two observations’ feature values.

    Attributes
    ----------
    candidate_thresholds_ : dict mapping features to list of thresholds
    """

    def fit(self, X, y):
        """ Finds all candidate split thresholds for each feature.

        Parameters
        ----------
        X : pandas DataFrame with observations, X.columns used as feature names
        y : pandas Series with labels

        Returns
        -------
        self
        """
        X_y = X.join(y)
        self.candidate_thresholds_ = {}
        for j in X.columns:
            thresholds = []
            sorted_X_y = X_y.sort_values([j, y.name])  # Sort by feature value, then by label
            prev_feature_val, prev_label = sorted_X_y.iloc[0][j], sorted_X_y.iloc[0][y.name]
            for idx, row in sorted_X_y.iterrows():
                curr_feature_val, curr_label = row[j], row[y.name]
                if (curr_label != prev_label and
                        not math.isclose(curr_feature_val, prev_feature_val)):
                    thresh = (prev_feature_val + curr_feature_val) / 2
                    thresholds.append(thresh)
                prev_feature_val, prev_label = curr_feature_val, curr_label
            self.candidate_thresholds_[j] = thresholds
        return self

    def transform(self, X):
        """ Binarize numerical features using candidate thresholds.

        Parameters
        ----------
        X : pandas DataFrame with observations, X.columns used as feature names

        Returns
        -------
        Xb : pandas DataFrame that is the result of binarizing X
        """
        check_is_fitted(self)
        Xb = pd.DataFrame()
        for j in X.columns:
            for threshold in self.candidate_thresholds_[j]:
                binary_test_name = "{}<={}".format(j, threshold)
                Xb[binary_test_name] = (X[j] <= threshold)
        Xb.replace({"False": 0, "True": 1}, inplace=True)
        return Xb


class Linear_Separator():
    """
    Assumes class labels are -1 and +1, and finds a hyperplane (a, c) such that a'x^i + 1 <= c iff y^i = -1,
                                                                                a'x^i - 1 >= c iff y^i = 1,
    Attempt to find the hyperplane using the following methods (in order):
        normalized l1-SVM MIP formulation
        Lagrangian dual of hard-margin linear SVM
        reduced dataset Lagrangian dual of hard-margin linear SVM
        generic hard margin hyperplane
        any generic hyperplane
        random hyperplane
    """

    def __init__(self):
        self.a_v = 0
        self.c_v = 0
        self.hp_size = 0

    def Lin_SVM_fit(self, data):
        # print('\nFITTING\n')
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
        data = data.drop(common_points_L | common_points_R)
        B = len(feature_set)

        m = Model("MIP SVM split normalized")
        u = m.addVars(feature_set, vtype=GRB.BINARY, name='u')
        err = m.addVars(data.index, vtype=GRB.CONTINUOUS, lb=0, name='err')
        b = m.addVar(vtype=GRB.CONTINUOUS, name='b')
        w_pos = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=0, name='w_pos')
        w_neg = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=0, name='w_neg')

        # m.setObjective((1 / 2) * quicksum((w_pos[f] - w_neg[f]) * (w_pos[f] - w_neg[f]) for f in feature_set) +
        #               err.sum(), GRB.MINIMIZE)
        m.setObjective(quicksum(w_pos[f] + w_neg[f] for f in feature_set) + err.sum(), GRB.MINIMIZE)
        m.addConstrs(data.at[i, 'svm'] * (quicksum((w_pos[f] - w_neg[f]) * data.at[i, f] for f in feature_set) + b)
                     >= 1 - err[i] for i in data.index)
        m.addConstr(u.sum() <= B)
        m.addConstrs(w_pos[f] <= u[f] for f in feature_set)
        m.addConstrs(w_neg[f] <= u[f] for f in feature_set)
        m.addConstrs(w_pos[f] - w_neg[f] >= 0 for f in feature_set)
        m.addConstr(w_neg.sum() <= 1)
        m.addConstr(w_pos.sum() <= 1)

        m.Params.LogToConsole = 0
        m.optimize()

        a_v = {f: w_pos[f].X - w_neg[f].X for f in feature_set}
        c_v = b.X
        u_dict = {f: u[f].X for f in feature_set if abs(a_v[f]) > 10**(-8)}
        self.a_v, self.c_v = a_v, c_v
        self.hp_size = sum(u_dict.values())

        return self

    def SVM_fit(self, data):
        feature_set = [f for f in data.columns if f != 'svm']
        left_index_set = [i for i in data.index if data.at[i, 'svm'] == -1]
        right_index_set = [i for i in data.index if data.at[i, 'svm'] == +1]
        try:
            m = Model("SVM")
            m.Params.LogToConsole = 0
            m.Params.NumericFocus = 3 # Prevents Gurobi from returning status code 12 (NUMERIC)
            alpha = m.addVars(data.index, lb=0, ub=GRB.INFINITY)
            w = m.addVars(feature_set, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.setObjective(alpha.sum() - (1/2)*quicksum(w[f]*w[f] for f in feature_set), GRB.MAXIMIZE)
            m.addConstrs((w[f] == quicksum(alpha[i]*data.at[i, 'svm']*data.at[i, f] for i in data.index)
                          for f in feature_set))
            m.addConstr(quicksum(alpha[i]*data.at[i, 'svm'] for i in data.index) == 0)
            m.optimize()
            # Any i with positive alpha[i] works
            for i in data.index:
                if alpha[i].X > m.Params.FeasibilityTol:
                    b = data.at[i, 'svm'] - sum(w[f].X*data.at[i, f] for f in feature_set)
                    break
            self.a_v = {f: w[f].X for f in feature_set}
            self.c_v = -b # Must flip intercept because of how QP was setup
            return self
        except Exception:
            # If QP fails to solve, just return any separating hyperplane
            m = Model("separating hyperplane")
            m.Params.LogToConsole = 1
            w = m.addVars(feature_set, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            b = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.setObjective(0, GRB.MINIMIZE)
            m.addConstrs((quicksum(w[f]*data.at[i, f] for f in feature_set) + 1 <= b for i in left_index_set))
            m.addConstrs((quicksum(w[f]*data.at[i, f] for f in feature_set) - 1 >= b for i in right_index_set))
            m.optimize()
            self.a_v = {f: w[f].X for f in feature_set}
            self.c_v = b.X
            return self


def model_results(model, tree):
    # Print assigned branching, classification, and pruned nodes of tree

    # Print tree node assignments (branching hyperplane weights, class, pruned)
    for v in tree.V:
        if model.P[v].x > 0.5:
            for k in model.classes:
                if model.W[v, k].x > 0.5:
                    print('Vertex ' + str(v) + ' class ' + str(k))
        elif model.P[v].x < 0.5 and model.B[v].x > 0.5:
            print('Vertex ' + str(v) + ' branching', tree.a_v[v], tree.c_v[v])
        elif model.P[v].x < 0.5 and model.B[v].x < 0.5:
            print('Vertex ' + str(v) + ' pruned')

    # Print datapoint paths through tree
    """for i in sorted(model.data.index):
        path = [0]
        for v in tree.V:
            if model.Q[i, v].x > 0.5:
                path.append(v)
                if model.S[i, v].x > 0.5:
                    print('Datapoint ' + str(i) + ' correctly assigned class ' + str(model.data.at[i, model.target])
                         + ' at ' + str(v) + '. Path: ', path)
                for k in model.data[model.target].unique():
                    if model.W[v, k].x > 0.5:
                        print('datapoint ' + str(i) + ' incorrectly assigned class ' + str(k)
                              + ' at ' + str(v) + '. Path: ', path)"""


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
    # Results dictionary for datapoint assignments and path through tree
    acc = 0
    results = {i: [None, []] for i in data.index}

    for i in data.index:
        v = 0
        while results[i][0] is None:
            results[i][1].append(v)
            if v in tree.branch_nodes:
                a_v, c_v = tree.a_v[v], tree.c_v[v]
                v = tree.LC[v] if sum(a_v[f] * data.at[i, f] for f in data.columns if f != 'target') <= c_v else tree.RC[v]
            elif v in tree.class_nodes:
                results[i][0] = tree.class_nodes[v]
                if results[i][0] == data.at[i, target]:
                    acc += 1
                    results[i].append('correct')
                else:
                    results[i].append('incorrect')
            else:
                results[i][0] = 'ERROR'

    return acc, results


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

