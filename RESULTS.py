import networkx as nx
import csv
import matplotlib.pyplot as plt


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
