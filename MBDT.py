import time
import pandas as pd
import networkx as nx
import random
from gurobipy import *
import UTILS


class MBDT:

    def __init__(self, data, tree, modeltype, time_limit, target, warmstart, modelextras, log=None):
        """"
        Parameters
        data: training data
        tree: input decision tree object of prespecified user height
        modeltype: modeltype to use for connectivity and optimization
        time_limit: gurobi time limit in seconds
        target: target column of training data
        warm_start: dictionary warm start values
        model_extras: list of modeltype extras
        """
        self.data = data
        self.tree = tree
        self.modeltype = modeltype
        self.time_limit = time_limit
        self.target = target
        self.warmstart = warmstart
        self.modelextras = modelextras
        self.log = log

        print('Model: ' + str(self.modeltype))
        # Feature, Class and Index Sets
        self.classes = data[target].unique()
        self.featureset = [col for col in self.data.columns.values if col != target]
        self.datapoints = data.index
        self.dataname = data.name

        """ Decision Variables """
        self.B = 0
        self.W = 0
        self.P = 0
        self.S = 0
        self.Q = 0
        # VIS Weights
        self.vis_weight = {i: 0 for i in self.datapoints}
        # CUT model separation constraints
        self.cut_constraint = 0
        self.single_terminal = 0

        """ Separation Procedure """
        sep_dict = {'FRAC-ALL': 'all', 'FRAC-FF': 'first found', 'FRAC-MV': 'most violating'}
        self.rootcuts = False
        self.eps = 0
        self.cut_type = self.modeltype[5:]
        if len(self.cut_type) == 0:
            self.cut_type = 'GRB'
        if ('ALL' in self.cut_type) or ('FF' in self.cut_type) or ('MV' in self.cut_type):
            if 'FRAC' not in self.cut_type: self.cut_type = 'FRAC-'+self.cut_type
            if 'ROOT' in self.cut_type:
                self.rootcuts = True
            print('User fractional separation cuts, type: ' + str(sep_dict[self.cut_type]) + ', root: ' + str(self.rootcuts))
        elif 'UF' in self.cut_type:
            print('ALL integral connectivity constraints')
        elif 'GRB' in self.cut_type:
            print('GRB lazy = 3 connectivity constraints')
        elif 'FRAC' in self.cut_type and len(self.cut_type) == 4:
            print('Need fractional separation type!!')
            return
        else:
            print('Need separation procedure!!')
            return

        """ Model extras """
        self.regularization = 'None'
        self.max_features = 'None'
        self.HP_time = 0

        """ Gurobi Optimization Parameters """
        self.model = Model(f'{self.modeltype}')
        self.model.Params.TimeLimit = time_limit
        self.model.Params.LogToConsole = 0
        self.model.Params.Threads = 1  # use one thread for testing purposes
        self.model.Params.LazyConstraints = 1
        self.model.Params.PreCrush = 1
        if self.log is not None:
            self.model.Params.LogFile = self.log + '.txt'

        """ Model callback metrics """
        self.model._septime, self.model._sepnum, self.model._sepcuts, self.model._sepavg = 0, 0, 0, 0
        self.model._vistime, self.model._visnum, self.model._viscuts = 0, 0, 0
        self.model._rootcuts, self.model._eps = self.rootcuts, self.eps

    ##############################################
    # MIP Model Formulation
    ##############################################
    def formulation(self):
        """
        Formulation of MIP modeltype with connectivity constraints according to modeltype type chosen by user
        returns optimal solution of model
        solved with gurobi
        """

        """ Decision Variables """
        # Branching vertex
        self.B = self.model.addVars(self.tree.V, vtype=GRB.BINARY, name='B')
        # Classification vertex
        self.W = self.model.addVars(self.tree.V, self.classes, vtype=GRB.BINARY, name='W')
        # Pruned vertex
        self.P = self.model.addVars(self.tree.V, vtype=GRB.CONTINUOUS, lb=0, name='P')
        # Datapoint terminal vertex
        self.S = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='S')
        # Datapoint selected vertices in root-terminal path
        self.Q = self.model.addVars(self.datapoints, self.tree.V, vtype=GRB.BINARY, name='Q')

        """ Model Objective and Constraints """
        # Objective: Maximize the number of correctly classified datapoints
        # Max sum(S[i,v], i in I, v in V\1)
        self.model.setObjective(quicksum(self.S[i, v] for i in self.datapoints for v in self.tree.V if v != 0),
                                GRB.MAXIMIZE)

        # Pruned vertices not assigned to class
        # P[v] = sum(W[v,k], k in K) for v in V
        self.model.addConstrs(self.P[v] == self.W.sum(v, '*')
                              for v in self.tree.V)

        # Vertices must be branched, assigned to class, or pruned
        # B[v] + sum(P[u], u in path[v]) = 1 for v in V
        self.model.addConstrs(self.B[v] + quicksum(self.P[u] for u in self.tree.path[v]) == 1
                              for v in self.tree.V)

        # Cannot branch on leaf vertex
        # B[v] = 0 for v in L
        for v in self.tree.L:
            self.B[v].ub = 0

        # Terminal vertex of correctly classified datapoint matches datapoint class
        # S[i,v] <= W[v,k=y^i] for v in V, for i in I
        for v in self.tree.V:
            self.model.addConstrs(self.S[i, v] <= self.W[v, self.data.at[i, self.target]]
                                  for i in self.datapoints)

        # If v not branching then all datapoints sent to left child
        # for v in self.tree.B:
        # self.model.addConstrs(self.Q[i, self.tree.RC[v]] <= self.B[v] for i in self.datapoints)

        # Each datapoint has at most one terminal vertex
        self.model.addConstrs(self.S.sum(i, '*') <= 1
                              for i in self.datapoints)

        # Lazy feasible path constraints (for fractional separation procedure)
        if ('GRB' in self.cut_type) or ('FRAC' in self.cut_type):
            # terminal vertex of datapoint must be in reachable path
            if 'CUT1' in self.modeltype:
                self.cut_constraint = self.model.addConstrs(self.S[i, v] <= self.Q[i, c]
                                                            for i in self.datapoints
                                                            for v in self.tree.V if v != 0
                                                            for c in self.tree.path[v][1:])
            # terminal vertex of datapoint must be in reachable path for vertex and all children
            elif 'CUT2' in self.modeltype:
                self.cut_constraint = self.model.addConstrs(self.S[i, v] + quicksum(self.S[i, u]
                                                                                    for u in self.tree.child[v])
                                                            <= self.Q[i, c]
                                                            for i in self.datapoints
                                                            for v in self.tree.V if v != 0
                                                            for c in self.tree.path[v][1:])
            for i in self.datapoints:
                for v in self.tree.V:
                    if v == 0: continue
                    for c in self.tree.path[v][1:]:
                        self.cut_constraint[i, v, c].lazy = 3

        # All feasible path constraints upfront
        elif 'ALL' in self.cut_type:
            for i in self.datapoints:
                for v in self.tree.V:
                    if v == 0: continue
                    # terminal vertex of datapoint must be in reachable path
                    if 'CUT1' in self.modeltype:
                        self.model.addConstrs(self.S[i, v] <= self.Q[i, c] for c in self.tree.path[v][1:])
                    # terminal vertex of datapoint must be in reachable path for vertex and all children
                    elif 'CUT2' in self.modeltype:
                        self.model.addConstrs(self.S[i, v] + quicksum(self.S[i, u] for u in self.tree.child[v]) <=
                                              self.Q[i, c] for c in self.tree.path[v][1:])

        """ Pass to Model DV for Callback / Optimization Purposes """
        self.model._B = self.B
        self.model._W = self.W
        self.model._S = self.S
        self.model._P = self.P
        self.model._Q = self.Q
        self.model._vis_weight, self.model._cut_type = self.vis_weight, self.cut_type
        self.model._data, self.model._tree = self.data, self.tree
        self.model._featureset, self.model._target = self.featureset, self.target

    ##############################################
    # Model Optimization / Callbacks
    ##############################################
    def optimization(self):
        self.model.optimize(MBDT.callbacks)

    @staticmethod
    def callbacks(model, where):
        """
        Gurobi Optimization
        At Feasible Solutions verify branching is valid for datapoints
            add VIS cuts
        In Branch and Bound Tree check path feasibility for datapoints
            add fractional separation cuts
        """
        # Model Termination
        if where == GRB.Callback.MIP:
            if abs(model.cbGet(GRB.Callback.MIP_OBJBST) -
                   model.cbGet(GRB.Callback.MIP_OBJBND)) < model.Params.FeasibilityTol:
                model.terminate()

        # Add VIS Cuts at Branching Nodes of Feasible Solution
        if where == GRB.Callback.MIPSOL:
            model._visnum += 1
            start = time.perf_counter()
            B = model.cbGetSolution(model._B)
            Q = model.cbGetSolution(model._Q)
            P = model.cbGetSolution(model._P)
            vis_weights = model._vis_weight

            for v in model._tree.B:
                if B[v] < 0.5: continue
                if P[v] > .5: continue
                # Define L_v(I), R_v(I) for each v in B
                Lv_I = set()  # set of i: q^i_l(v) = 1
                Rv_I = set()  # set of i: q^i_r(v) = 1
                for i in model._data.index:
                    if Q[i, model._tree.LC[v]] > 0.5:
                        Lv_I.add(i)
                    elif Q[i, model._tree.RC[v]] > 0.5:
                        Rv_I.add(i)
                # Test for VIS of B_v(Q)
                print('Test for VIS at', v, model._visnum)
                VIS = MBDT.VIS(model._data, model._featureset, Lv_I, Rv_I, vis_weight=vis_weights)
                if VIS is None:
                    continue
                # If VIS Found, add cut
                (B_v_left, B_v_right) = VIS
                model.cbLazy(quicksum(model._Q[i, model._tree.LC[v]] for i in B_v_left) +
                             quicksum(model._Q[i, model._tree.LC[v]] for i in B_v_right) <=
                             len(B_v_left) + len(B_v_right) - 1)
                model._viscuts += 1
            model._vistime += time.perf_counter() - start

        # Add Feasible Path for Datapoints Cuts at Fractional Point in Branch and Bound Tree
        if (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            if ('ALL' in model._cut_type) or ('GRB' in model._cut_type): pass
            print('in bb tree')
            start = time.perf_counter()
            # Only add cuts at root-node of branch and bound tree
            if model._rootcuts:
                if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
            model._sepnum += 1
            q_val = model.cbGetNodeRel(model._Q)
            s_val = model.cbGetNodeRel(model._S)
            # Add all violating cuts
            if 'A' in model._cut_type:
                if 'CUT1' in model.ModelName:
                    for (i, v) in s_val.keys():
                        for c in model._tree.path[v][1:]:
                            if s_val[i, v] - q_val[i, c] > 10 ** -model._eps:
                                model.cbCut(model._S[i, v] <= model._Q[i, c])
                                model._sepcuts += 1
                elif 'CUT2' in model.ModelName:
                    for (i, v) in s_val.keys():
                        for c in model._tree.path[v][1:]:
                            if s_val[i, v] + sum(s_val[i, u] for u in model._tree.child[v]) - \
                                    q_val[i, c] > 10 ** -model._eps:
                                model.cbCut(model._S[i, v] +
                                            quicksum(model._S[i, u] for u in model._tree.child[v]) <=
                                            model._Q[i, c])
                                model._sepcuts += 1
            # Add first found violating cut
            elif 'FF' in model._cut_type:
                if 'CUT1' in model.ModelName:
                    for (i, v) in s_val.keys():
                        for c in model._tree.path[v][1:]:
                            if s_val[i, v] - q_val[i, c] > 10 ** -model._eps:
                                model.cbCut(model._S[i, v] <= model._Q[i, c])
                                model._sepcuts += 1
                                break
                elif 'CUT2' in model.ModelName:
                    for (i, v) in s_val.keys():
                        for c in model._tree.path[v][1:]:
                            if s_val[i, v] + sum(s_val[i, u] for u in model._tree.child[v]) - \
                                    q_val[i, c] > 10 ** -model._eps:
                                model.cbCut(model._S[i, v] +
                                            quicksum(model._S[i, u] for u in model._tree.child[v]) <=
                                            model._Q[i, c])
                                model._sepcuts += 1
                                break
            # Add most violating cut
            elif 'MV' in model._cut_type:
                if 'CUT1' in model.ModelName:
                    for (i, v) in s_val.keys():
                        for c in model._tree.path[v][1:]:
                            if (s_val[i, v] - q_val[i, c] > 10 ** -model._eps and
                                    s_val[i, v] - q_val[i, c] == max(
                                        s_val[i, v] - q_val[i, d] for d in model._tree.path[v][1:])):
                                model.cbCut(model._S[i, v] <= model._Q[i, c])
                                model._sepcuts += 1
                                break
                elif 'CUT2' in model.ModelName:
                    for (i, v) in s_val.keys():
                        for c in model._tree.path[v][1:]:
                            if (s_val[i, v] + sum(s_val[i, u] for u in model._tree.child[v]) - q_val[
                                i, c] > 10 ** -model._eps
                                    and s_val[i, v] + sum(s_val[i, u] for u in model._tree.child[v]) - q_val[i, c] ==
                                    max(s_val[i, v] + sum(s_val[i, u] for u in model._tree.child[v]) - q_val[i, d]
                                        for d in model._tree.path[v][1:])):
                                model.cbCut(model._S[i, v] +
                                            quicksum(model._S[i, u] for u in model._tree.child[v]) <=
                                            model._Q[i, c])
                                model._sepcuts += 1
                                break
            model._septime += (time.perf_counter() - start)

    ##############################################
    # Find Valid Infeasible Subsystem (VIS) of Model
    ##############################################
    @staticmethod
    def VIS(data, feature_set, Lv_I, Rv_I, vis_weight):
        """
        Find a minimal set of points that cannot be linearly separated by a split (a_v, c_v).
        Use the support of Farkas dual (with heuristic objective) of the feasible primal LP to identify VIS of primal
        Primal is B_v(Q) : a_v*x^i + 1 <= c_v for 1 for i in L_v(I) := {i in I : q^i_l(v) = 1}
                           a_v*x^i - 1 <= c_v for 1 for i in R_v(I) := {i in I : q^i_r(v) = 1}
        Parameters
        data : dataframe of shape (I, F)
        Lv_I : list of I s.t. q^i_l(v) = 1
        Rv_I : list of I s.t. q^i_r(v) = 1
        vis_weight : ndarray of shape (N,), default=None
            Objective coefficients of Farkas dual

        Returns
        (B_v_left, B_v_right) : two lists of left and right datapoint indices in the VIS of B_v(Q)
        """

        if vis_weight is None:
            vis_weight = {i: 0 for i in data.index}

        if (len(Lv_I) == 0) or (len(Rv_I) == 0):
            return None
        # Remove any points in each index set whose feature set are equivalent
        common_points_L, common_points_R = set(), set()
        for x in Lv_I:
            for y in Rv_I:
                if data.loc[x, feature_set].equals(data.loc[y, feature_set]):
                    common_points_L.add(x)
                    common_points_R.add(y)
        Lv_I -= common_points_L
        Rv_I -= common_points_R
        data = data.drop(common_points_L | common_points_R)

        # VIS Dual Model
        VIS_model = Model("VIS Dual")
        VIS_model.Params.LogToConsole = 0

        # VIS Dual Variables
        lambda_L = VIS_model.addVars(Lv_I, vtype=GRB.CONTINUOUS, lb=0, name='lambda_L')
        lambda_R = VIS_model.addVars(Rv_I, vtype=GRB.CONTINUOUS, lb=0, name='lambda_R')

        # VIS Dual Constraints
        VIS_model.addConstrs(
            quicksum(lambda_L[i] * data.at[i, j] for i in Lv_I) ==
            quicksum(lambda_R[i] * data.at[i, j] for i in Rv_I)
            for j in feature_set)
        VIS_model.addConstr(lambda_L.sum() == 1)
        VIS_model.addConstr(lambda_R.sum() == 1)

        # VIS Dual Objective
        VIS_model.setObjective(quicksum(vis_weight[i] * lambda_L[i] for i in Lv_I) +
                               quicksum(vis_weight[i] * lambda_R[i] for i in Rv_I), GRB.MINIMIZE)

        # Optimize
        VIS_model.optimize()

        # Infeasiblity implies B_v(Q) is valid for all I in L_v(I), R_v(I)
        # i.e. each i is correctly sent to left, right child (linearly separable points)
        if VIS_model.Status == GRB.INFEASIBLE:
            print('none found')
            return None
        print('found VIS')
        lambda_L_sol = VIS_model.getAttr('X', lambda_L)
        lambda_R_sol = VIS_model.getAttr('X', lambda_R)

        B_v_left = []
        B_v_right = []
        for i in Lv_I:
            if lambda_L_sol[i] > VIS_model.Params.FeasibilityTol:
                B_v_left.append(i)
                vis_weight[i] += 1
        for i in Rv_I:
            if lambda_R_sol[i] > VIS_model.Params.FeasibilityTol:
                B_v_right.append(i)
                vis_weight[i] += 1
        print(B_v_left, B_v_right)
        return (B_v_left, B_v_right)

    ##############################################
    # Warm Start Model
    ##############################################
    def warm_start(self):
        """
        Warm start tree with random (a,c) and classes
        Generate random tree from UTILS.random_tree()
        Find feasible path and terminal vertex for each datapoint through tree
        Update according decision variable values
        """
        # Generate random tree assignments if none were given
        if not self.warmstart['use']:
            return self
        if self.warmstart['values'] is None:
            warm_start_values = UTILS.random_tree(self.tree, self.data, self.target)
            print('Warm starting model with randomly generated tree')
        else:
            warm_start_values = self.warmstart['values']
            print('Warm starting model with passed values')
        # Check tree assignments are valid
        if not UTILS.tree_check(warm_start_values['tree']):
            print('Invalid Tree!!')

        # Retrieve node assignments
        class_nodes = nx.get_node_attributes(warm_start_values['tree'].DG_prime, 'class')
        branching_nodes = nx.get_node_attributes(warm_start_values['tree'].DG_prime, 'branching')
        pruned_nodes = nx.get_node_attributes(warm_start_values['tree'].DG_prime, 'pruned')

        # Warm start node assignment decision variables (B, W, P)
        for v in class_nodes:
            for k in self.classes:
                if warm_start_values['tree'].DG_prime.nodes[v]['class'] == k:
                    self.W[v, k].Start = 1.0
                else:
                    self.W[v, k].Start = 0
            self.P[v].Start = 1.0
            self.B[v].Start = 0
        for v in branching_nodes:
            self.B[v].Start = 1.0
            for k in self.classes:
                self.W[v, k].Start = 0
            self.P[v].Start = 0
        for v in pruned_nodes:
            self.P[v].Start = 0
            self.B[v].Start = 0
            for k in self.classes:
                self.W[v, k].Start = 0

        # Warm start datapoint selected source-terminal nodes decision variables (S, Q)
        for i in self.datapoints:
            for v in warm_start_values['tree'].V:
                if v == warm_start_values['results'][i][1][-1] and 'correct' in warm_start_values['results'][i]:
                    self.S[i, v].Start = 1.0
                elif v == warm_start_values['results'][i][1][-1]:
                    self.S[i, v].Start = 0.0
                else:
                    self.S[i, v].Start = 0.0
                if v in warm_start_values['results'][i][1]:
                    self.Q[i, v].Start = 1.0
                else:
                    self.Q[i, v].Start = 0.0

    ##############################################
    # Model Extras
    ##############################################
    def extras(self):
        # number of maximum branching nodes
        if any((match := elem).startswith('max_features') for elem in self.modelextras):
            self.max_features = int(re.sub("[^0-9]", "", match))
            print('No more than ' + str(self.max_features) + ' feature(s) used')
            self.model.addConstr(quicksum(self.B[v] for v in self.tree.B) <= self.max_features)

        # exact number of branching nodes
        if any((match := elem).startswith('num_features') for elem in self.modelextras):
            self.max_features = int(re.sub("[^0-9]", "", match))
            print(str(self.max_features) + ' feature(s) used')
            self.model.addConstr(quicksum(self.B[v] for v in self.tree.B) == self.max_features)

        # regularization
        if any((match := elem).startswith('regularization') for elem in self.modelextras):
            self.regularization = int(re.sub("[^0-9]", "", match))
            print('Regularization: ' + str(self.regularization) + ' datapoints required for classification vertices')
            self.model.addConstrs(quicksum(self.S[i, v] for i in self.datapoints) >= self.regularization * self.P[v]
                                  for v in self.tree.V)

    ##############################################
    # Assign Nodes of Tree from Model Solution
    ##############################################
    def assign_tree(self):
        """
        Assign nodes of tree from model solution
        Assign class k to v when P[v].x = 1, W[v, k].x = 1
        Assign pruned v to when P[v].x = 0, B[v].x = 0
        Define (a_v, c_v) for v when P[v].x = 0, B[v].x = 1
            Use hard margin linear SVM on B_v(Q) to find (a_v, c_v)
        """
        start = time.perf_counter()
        # clear any existing node assignments
        for v in self.tree.DG_prime.nodes():
            if 'class' in self.tree.DG_prime.nodes[v]:
                del self.tree.DG_prime.nodes[v]['class']
            if 'branching' in self.tree.DG_prime.nodes[v]:
                del self.tree.DG_prime.nodes[v]['branching']
            if 'pruned' in self.tree.DG_prime.nodes[v]:
                del self.tree.DG_prime.nodes[v]['pruned']

        # Retrieve solution values
        try:
            B_sol = self.model.getAttr('X', self.B)
            W_sol = self.model.getAttr('X', self.W)
            P_sol = self.model.getAttr('X', self.P)
            Q_sol = self.model.getAttr('X', self.Q)
        # If no incumbent was found, then predict arbitrary class at root node
        except GurobiError:
            self.tree.DG_prime.nodes[0]['class'] = random.choice(self.classes)
            for v in self.tree.V:
                if v == 0: continue
                self.tree.DG_prime.nodes[v]['pruned'] = 0
            return

        for v in self.tree.V:
            # Assign class k to classification nodes
            if P_sol[v] > 0.5:
                for k in self.classes:
                    if W_sol[v, k] > 0.5:
                        self.tree.DG_prime.nodes[v]['class'] = k
            # Assign no class or branching rule to pruned nodes
            elif P_sol[v] < 0.5 and B_sol[v] < 0.5:
                self.tree.DG_prime.nodes[v]['pruned'] = 0
            # Define (a_v, c_v) on branching nodes
            elif P_sol[v] < 0.5 and B_sol[v] > 0.5:
                # Lv_I, Rv_I index sets of observations sent to left, right child vertex of branching vertex v
                # svm_y maps Lv_I to -1, Rv_I to +1 for training hard margin linear SVM
                Lv_I, Rv_I = [], []
                svm_y = {i: 0 for i in self.datapoints}
                for i in self.datapoints:
                    if Q_sol[i, self.tree.LC[v]] > 0.5:
                        Lv_I.append(i)
                        svm_y[i] = -1
                    elif Q_sol[i, self.tree.RC[v]] > 0.5:
                        Rv_I.append(i)
                        svm_y[i] = +1
                print(svm_y)
                # Find (a_v, c_v) for corresponding Lv_I, Rv_I
                # If |Lv_I| = 0: (a_v, c_v) = (0, -1) sends all points to the right
                if len(Lv_I) == 0:
                    self.tree.a_v[v] = {f: 0 for f in self.featureset}
                    self.tree.c_v[v] = -1
                # If |Rv_I| = 0: (a_v, c_v) = (0, 1) sends all points to the left
                elif len(Rv_I) == 0:
                    self.tree.a_v[v] = {f: 0 for f in self.featureset}
                    self.tree.c_v[v] = 1
                # Find separating hyperplane according to Lv_I, Rv_I index sets
                else:
                    data_svm = self.data.loc[Lv_I + Rv_I, self.data.columns != self.target]
                    data_svm['svm'] = pd.Series(svm_y)
                    print(data_svm.head(5))
                    svm = UTILS.Linear_Separator()
                    svm.SVM_fit(data_svm)
                    self.tree.a_v[v], self.tree.c_v[v] = svm.a_v, svm.c_v
                self.tree.DG_prime.nodes[v]['branching'] = (self.tree.a_v[v], self.tree.c_v[v])
        self.HP_time = time.perf_counter() - start
