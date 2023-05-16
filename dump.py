"""
def random_tree(tree, training_data, target):
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
        TD_tree = TD_rand_tree(tree, training_data, target)
        TD_acc, TD_results = data_predict(TD_tree, training_data, target)
        if TD_acc > TD_best_acc:
            TD_best_acc = TD_acc
            TD_best_results = TD_results
            best_TD_tree = TD_tree

    for i in range(50):
        BU_tree = BU_rand_tree(tree, training_data, target)
        BU_acc, BU_results = data_predict(BU_tree, training_data, target)
        if BU_acc > BU_best_acc:
            BU_best_acc = BU_acc
            BU_best_results = BU_results
            best_BU_tree = BU_tree
    if TD_best_acc > BU_best_acc:
        return {'tree': best_TD_tree, 'results': TD_best_results}
    else:
        return {'tree': best_BU_tree, 'results': BU_best_results}


def TD_rand_tree(tree, training_data, target):
    # Clear any existing node assignments
    for v in tree.V:
        if 'class' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['class']
        if 'branching' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['branching']
        if 'pruned' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['pruned']
    classes = training_data[target].unique()
    feature_set = [col for col in training_data.columns if col != target]

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


def BU_rand_tree(tree, training_data, target):
    # Clear any existing node assignments
    for v in tree.V:
        if 'class' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['class']
        if 'branching' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['branching']
        if 'pruned' in tree.DG_prime.nodes[v]:
            del tree.DG_prime.nodes[v]['pruned']

    classes = training_data[target].unique()
    feature_set = [col for col in training_data.columns if col != target]

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
        elif hp_type == 'double-cube':
            m = Model("MIP SVM normalized")
            u = m.addVars(feature_set, vtype=GRB.BINARY, name='u')
            err = m.addVars(training_data.index, vtype=GRB.CONTINUOUS, lb=0, name='err')
            b = m.addVar(vtype=GRB.CONTINUOUS, name='b')
            w = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=-1, name='w')
            if hp_obj == 'linear':
                m.setObjective(w.sum() + epsilon*err.sum(), GRB.MINIMIZE)
            elif hp_obj == 'quadratic':
                m.setObjective((1 / 2) * quicksum(w[f]*w[f] for f in feature_set) + epsilon * err.sum(), GRB.MINIMIZE)
            elif hp_obj == 'rank':
                m.setObjective(u.sum() + 10 * err.sum(), GRB.MINIMIZE)
            m.addConstrs(training_data.at[i, 'svm'] * (quicksum(w[f] * training_data.at[i, f] for f in feature_set) + b)
                         >= 1 - err[i] for i in training_data.index)
            m.addConstr(quicksum(u[f] for f in feature_set) <= B-1)
            m.addConstrs(w[f] <= u[f] for f in feature_set)
            m.addConstr(w.sum() <= 1)
            m.addConstr(w.sum() >= 0)
            m.addConstrs(-1*(w[f]+1) >= -1*u[f] for f in feature_set)

            m.Params.LogToConsole = 1
            m.optimize()
            for f in feature_set:
                print(f, w[f], u[f])
            a_v = {f: w[f].x for f in feature_set}
            c_v = b.x
            # u_dict = {f: u[f].x for f in feature_set}
            # print('u values', u_dict)
            # print('a_values', a_v)
            self.a_v, self.c_v = a_v, c_v


        try:
            # Find separating hyperplane using l1-SVM normalized MIP formulation
            m = Model("MIP SVM normalized")
            m.Params.LogToConsole = 0
            u = m.addVars(feature_set, vtype=GRB.BINARY, name='U')
            err = m.addVars(training_data.index, vtype=GRB.CONTINUOUS, lb=0, name='err')
            b_mip = m.addVar(vtype=GRB.CONTINUOUS, name='b')
            w = m.addVars(vtype=GRB.CONTINUOUS, lb=0, name='w')

            m.setObjective(u.sum()+C[0]*err.sum(), GRB.MINIMIZE)
            m.addConstrs(training_data.at[i, 'svm'] * (quicksum(w[f]*training_data.at[i, f] for f in feature_set) + b_mip) >= 1 - err[i]
                         for i in training_data.index)
            m.addConstrs(u.sum() <= B)
            m.addConstrs(w[f] <= u[f] for f in feature_set)
            m.addConstr(w.sum() <= 1)
            m.optimize()

        except Exception:
            # Find separating hyperplane by solving dual of Lagrangian of the standard hard margin linear SVM problem
            try:
                m = Model("HM_Linear_SVM")
                m.Params.LogToConsole = 0
                m.Params.NumericFocus = 3
                alpha = m.addVars(training_data.index, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
                W = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

                m.addConstrs(W[f] == quicksum(alpha[i] * training_data.at[i, 'svm'] * training_data.at[i, f] for i in training_data.index)
                             for f in feature_set)
                m.addConstr(quicksum(alpha[i] * training_data.at[i, 'svm'] for i in training_data.index) == 0)
                m.setObjective(alpha.sum() - (1 / 2) * quicksum(W[f] * W[f] for f in feature_set), GRB.MAXIMIZE)
                m.optimize()

                # Any i with positive alpha[i] works
                for i in training_data.index:
                    if alpha[i].x > m.Params.FeasibilityTol:
                        b = training_data.at[i, 'svm'] - sum(W[f].x * training_data.at[i, f] for f in feature_set)
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
                        convex_combo.addConstrs(quicksum(lambdas[i]*training_data.at[i, f] for i in Rv_I) == training_data.at[i, f]
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
                        convex_combo.addConstrs(quicksum(lambdas[i]*training_data.at[i, f] for i in Lv_I) == training_data.at[i, f]
                                                for f in feature_set)
                        convex_combo.addConstr(lambdas.sum() == 1)
                        convex_combo.setObjective(0, GRB.MINIMIZE)
                        convex_combo.optimize()
                        if convex_combo.Status != GRB.INFEASIBLE:
                            cc_R.add(i)

                    # Find noramlized max inner product of convex combinations
                    # to use as upper bound in dual of Lagrangian of soft margin SVM
                    margin_ub = GRB.INFINITY
                    inner_products = {item: np.inner(training_data.loc[item[0], feature_set],
                                              training_data.loc[item[1], feature_set])
                                      for item in list(combinations(cc_L|cc_R, 2))}
                    if inner_products:
                        margin_ub = max(inner_products.values()) / \
                                    min(len(cc_L | cc_R), np.linalg.norm(list(inner_products.values()), 2))

                    # Solve dual of Lagrangian of soft margin SVM
                    m = Model("SM_Linear_SVM")
                    m.Params.LogToConsole = 0
                    m.Params.NumericFocus = 3
                    alpha = m.addVars(training_data.index, vtype=GRB.CONTINUOUS, lb=0, ub=margin_ub)
                    W = m.addVars(feature_set, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

                    m.addConstrs(W[f] == quicksum(alpha[i] * training_data.at[i, 'svm'] * training_data.at[i, f] for i in training_data.index)
                                 for f in feature_set)
                    m.addConstr(quicksum(alpha[i] * training_data.at[i, 'svm'] for i in training_data.index) == 0)
                    m.setObjective(alpha.sum() - (1 / 2) * quicksum(W[f] * W[f] for f in feature_set), GRB.MAXIMIZE)
                    m.optimize()

                    # Any i with positive alpha[i] works
                    for i in training_data.index:
                        if alpha[i].x > m.Params.FeasibilityTol:
                            b = training_data.at[i, 'svm'] - sum(W[f].x * training_data.at[i, f] for f in feature_set)
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
                            quicksum(a_hyperplane[f] * training_data.at[i, f] for f in feature_set) + 1 <= c_hyperplane
                            for i in Lv_I)
                        gen_hyperplane.addConstrs(
                            quicksum(a_hyperplane[f] * training_data.at[i, f] for f in feature_set) - 1 >= c_hyperplane
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
                                quicksum(a_hyperplane[f] * training_data.at[i, f] for f in feature_set) <= c_hyperplane
                                for i in Lv_I)
                            gen_hyperplane.addConstrs(
                                quicksum(a_hyperplane[f] * training_data.at[i, f] for f in feature_set) >= c_hyperplane
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

"""