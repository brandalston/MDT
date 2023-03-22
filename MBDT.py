import time
import pandas as pd
import networkx as nx
from math import floor
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
        self.b_type = '2-Step'

        # Feature, Class and Index Sets
        self.classes = data[target].unique()
        self.featureset = [col for col in self.data.columns.values if col != target]
        self.datapoints = data.index

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
        self.rootcuts = False
        self.eps = 0
        self.cut_type = self.modeltype[5:]
        if len(self.cut_type) == 0:
            self.cut_type = 'GRB'
        if any(ele in self.cut_type for ele in ['FF', 'ALL', 'MV']):
            self.eps = 4

        """ Model extras """
        self.regularization = 'None'
        self.max_features = 'None'
        self.HP_time = 0
        self.HP_size = 0
        self.svm_branches = 0
        self.obj_func = 'N/A'

        """ Gurobi Optimization Parameters """
        self.model = Model(f'{self.modeltype}_SVM')
        self.model.Params.LogToConsole = 0
        self.model.Params.TimeLimit = time_limit
        self.model.Params.Threads = 1  # use one thread for testing purposes
        self.model.Params.LazyConstraints = 1
        self.model.Params.PreCrush = 1
        # Save Gurobi log to file
        if self.log:
            self.model.Params.LogFile = self.log
        """ Model callback metrics """
        self.model._septime, self.model._sepnum, self.model._sepcuts, self.model._sepavg = 0, 0, 0, 0
        self.model._vistime, self.model._visnum, self.model._viscuts = 0, 0, 0
        self.model._rootcuts, self.model._eps = self.rootcuts, self.eps

        """ Hyperplane Specifications 
        self.HP_obj, self.HP_rank = 'quadratic', len(self.featureset)
        if self.hp_info is not None:
            # print(f'Hyperplane Objective:', self.hp_info['objective'])
            self.HP_obj = self.hp_info['objective']
            if type(self.hp_info['rank']) is float:
                self.HP_rank = floor(hp_info['rank'] * len(self.featureset))
            elif hp_info['rank'] == 'full':
                self.HP_rank = len(self.featureset)
            else:
                self.HP_rank = len(self.featureset) - 1
            # print(f'Hyperplane Rank:', self.hp_rank) """

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
        # Pruned vertex
        self.P = self.model.addVars(self.tree.V, vtype=GRB.CONTINUOUS, lb=0, name='P')
        # Branching vertex
        self.B = self.model.addVars(self.tree.V, vtype=GRB.BINARY, name='B')
        # Classification vertex
        self.W = self.model.addVars(self.tree.V, self.classes, vtype=GRB.BINARY, name='W')
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

        # If v not branching then all datapoints sent to right child
        # for v in self.tree.B:
            # self.model.addConstrs(self.Q[i, self.tree.RC[v]] <= self.B[v] for i in self.datapoints)
            # self.model.addConstrs(self.Q[i, self.tree.LC[v]] <= self.B[v] for i in self.datapoints)

        # Each datapoint has at most one terminal vertex
        self.model.addConstrs(self.S.sum(i, '*') <= 1
                              for i in self.datapoints)

        # Lazy feasible path constraints (for fractional separation procedure)
        if any(ele in self.cut_type for ele in ['GRB', 'FF', 'ALL', 'MV']):
            # terminal vertex of datapoint must be in reachable path
            if 'CUT1' in self.modeltype:
                self.cut_constraint = self.model.addConstrs(
                    self.S[i, v] <= self.Q[i, c]
                    for i in self.datapoints
                    for v in self.tree.V if v != 0
                    for c in self.tree.path[v][1:])
            # terminal vertex of datapoint must be in reachable path for vertex and all children
            elif 'CUT2' in self.modeltype:
                self.cut_constraint = self.model.addConstrs(
                    self.S[i, v] + quicksum(self.S[i, u] for u in self.tree.child[v])
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
        elif 'UF' in self.cut_type:
            for v in self.tree.V:
                if v == 0: continue
                for i in self.datapoints:
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
                # print(f'Test for VIS at {v}, VIS test count: {model._visnum}')
                VIS = UTILS.VIS(model._data, Lv_I, Rv_I, vis_weight=model._vis_weight)

                # If VIS Found, add cut
                if VIS is None: continue
                # print(f'VIS Found at {v}, test count: {model._visnum}')
                (VIS_left, VIS_right) = VIS
                model.cbLazy(quicksum(model._Q[i, model._tree.LC[v]] for i in VIS_left) +
                             quicksum(model._Q[i, model._tree.RC[v]] for i in VIS_right) <=
                             len(VIS_left) + len(VIS_right) - 1)
                model._viscuts += 1
            model._vistime += time.perf_counter() - start

        # Add Feasible Path for Datapoints Cuts at Fractional Point in Branch and Bound Tree
        if (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            if ('UF' in model._cut_type) or ('GRB' in model._cut_type): return
            start = time.perf_counter()
            # Only add cuts at root-node of branch and bound tree
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
            model._sepnum += 1
            q_val = model.cbGetNodeRel(model._Q)
            s_val = model.cbGetNodeRel(model._S)
            # Add all violating cuts
            if 'ALL' in model._cut_type:
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
            self.tree.class_nodes[0] = random.choice(self.classes)
            for v in self.tree.V:
                if v == 0: continue
                self.tree.pruned_nodes[v] = 0
            return
        branched = []
        for v in self.tree.V:
            # Assign class k to classification nodes
            if P_sol[v] > 0.5:
                for k in self.classes:
                    if W_sol[v, k] > 0.5:
                        self.tree.class_nodes[v] = k
                        # print(f'{v} assigned class {k}')
            # Assign no class or branching rule to pruned nodes
            elif P_sol[v] < 0.5 and B_sol[v] < 0.5:
                self.tree.pruned_nodes[v] = 0
            # Define (a_v, c_v) on branching nodes
            elif P_sol[v] < 0.5 and B_sol[v] > 0.5:
                branched.append(v)
        for v in branched:
            # Lv_I, Rv_I index sets of observations sent to left, right child vertex of branching vertex v
            # svm_y maps Lv_I to -1, Rv_I to +1 for training hyperplane
            Lv_I, Rv_I = [], []
            svm = {i: 0 for i in self.datapoints}
            for i in self.datapoints:
                if Q_sol[i, self.tree.LC[v]] > 0.5:
                    Lv_I.append(i)
                    svm[i] = -1
                elif Q_sol[i, self.tree.RC[v]] > 0.5:
                    Rv_I.append(i)
                    svm[i] = +1
            # Find (a_v, c_v) for corresponding Lv_I, Rv_I
            # If |Lv_I| = 0: (a_v, c_v) = (0, -1) sends all points to the right
            if len(Lv_I) == 0:
                # print(f'all going right at {v}')
                self.tree.a_v[v] = {f: 0 for f in self.featureset}
                self.tree.c_v[v] = -1
            # If |Rv_I| = 0: (a_v, c_v) = (0, 1) sends all points to the left
            elif len(Rv_I) == 0:
                # print(f'all going left at {v}')
                self.tree.a_v[v] = {f: 0 for f in self.featureset}
                self.tree.c_v[v] = 1
            # Find separating hyperplane according to Lv_I, Rv_I index sets
            else:
                # print('BRANCHING!!')
                # print('branching at', v)
                data_svm = self.data.loc[Lv_I + Rv_I, self.data.columns != self.target]
                data_svm['svm'] = pd.Series(svm)
                svm = UTILS.Linear_Separator()
                svm.SVM_fit(data_svm)
                self.tree.a_v[v], self.tree.c_v[v] = svm.a_v, svm.c_v
                if svm.hp_size != 0:
                    self.svm_branches += 1
                    self.HP_size += svm.hp_size
            self.tree.branch_nodes[v] = (self.tree.a_v[v], self.tree.c_v[v])
        self.HP_time = time.perf_counter() - start
        # print(f'Hyperplanes found in {round(self.HP_time,4)}s. ({time.strftime("%I:%M %p", time.localtime())})\n')

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
        if any((match := elem).startswith('max_branch') for elem in self.modelextras):
            self.max_features = int(re.sub("[^0-9]", "", match))
            print('No more than ' + str(self.max_features) + ' branching vertices used')
            self.model.addConstr(quicksum(self.B[v] for v in self.tree.B) <= self.max_features)

        # exact number of branching nodes
        if any((match := elem).startswith('num_branch') for elem in self.modelextras):
            self.max_features = int(re.sub("[^0-9]", "", match))
            print(str(self.max_features) + ' branching vertices used')
            self.model.addConstr(quicksum(self.B[v] for v in self.tree.B) == self.max_features)

        # regularization
        if any((match := elem).startswith('regularization') for elem in self.modelextras):
            self.regularization = int(re.sub("[^0-9]", "", match))
            print('Regularization: ' + str(self.regularization) + ' datapoints required for classification vertices')
            self.model.addConstrs(quicksum(self.S[i, v] for i in self.datapoints) >= self.regularization * self.P[v]
                                  for v in self.tree.V)