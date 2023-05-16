#!/usr/bin/python
import os, time, getopt, sys, csv
import pandas as pd
from sklearn.model_selection import train_test_split
from MBDT import MBDT
from MBDT_ONE_STEP import MBDT_one_step
from TREE import TREE
import UTILS


def main(argv):
    # print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    rand_states = None
    model_extras = None
    file_out = None
    log_files = None
    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:e:w:l:",
                                   ["data_files=", "heights=", "time_limit=", "models=",
                                    "rand_states=", "results_file=", "model_extras=", "log_files"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--time_limit"):
            time_limit = int(arg)
        elif opt in ("-m", "--model"):
            modeltypes = arg
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-e", "--model_extras"):
            model_extras = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'VIS_calls', 'VIS_cuts', 'VIS_time', 'HP_time',
                       'FP_CB_Time', 'FP_Num_CB', 'FP_Cuts', 'Eps', 'Branch_Type']
    """summary_columns = ['Data', 'H', '|I|', '|F|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound',
                       'Model', 'Branch_Type',
                       'HP_Time', 'HP_Size', 'HP_Obj', 'HP_Rank',
                       'FP_CB_Time', 'FP_Num_CB', 'FP_Cuts', 'FP_Avg',
                       'VIS_CB_Time', 'VIS_Num_CB', 'VIS_Cuts',
                       'Eps', 'Time_Limit', 'Rand_State',
                       'Warm Start', 'Regularization', 'Max_Features']"""
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_E:' + str(model_extras) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()
    """ We assume the target column of dataset is labeled 'target'
    Change value at your discretion """
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                          'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        # if file in numerical_datasets: binarization = 'all-candidates'
        # else: binarization = False
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), binarization=None)
        for h in heights:
            for i in rand_states:
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                for modeltype in modeltypes:
                    print('\n'+str(modeltype) +
                          ', Dataset: ' + str(file) + ', H: ' + str(h) + ', ''Rand State: ' + str(i) +
                          '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    if log_files: log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(
                        modeltype) + '_T:' + str(time_limit) + '_Seed:' + str(i) + '_E:' + str(
                        model_extras)
                    else: log = None
                    # Generate tree and necessary structure information
                    tree_ws = TREE(h=h)
                    # Model with 75% training set and time limit
                    mbdt_ws = MBDT(data=model_set, tree=tree_ws, target=target, modeltype=modeltype,
                                   time_limit=time_limit, warmstart=False,
                                   modelextras=model_extras, log=log, log_to_console=0)
                    mbdt_ws.formulation()
                    mbdt_ws.model.update()
                    mbdt_ws.optimization()
                    mbdt_ws.assign_tree()

                    ws_acc, ws_assignments = UTILS.data_predict(tree=tree_ws, data=model_set, target=mbdt_ws.target)
                    print(tree_ws.branch_nodes)

                    """a_v_norm, c_v_norm, branch_norm = {}, {}, {}
                    for v in tree_ws.branch_nodes:
                        factor = 1.0 / sum(tree_ws.branch_nodes[v][0].values(),tree_ws.branch_nodes[v][1])
                        a_v_norm[v] = {k: v*factor for k, v in tree_ws.branch_nodes[v][0].items()}
                        c_v_norm[v] = tree_ws.branch_nodes[v][1]*factor
                        branch_norm[v] = (a_v_norm[v], c_v_norm[v])
                    print(branch_norm)
                    tree_ws_norm = TREE(h=h)
                    tree_ws_norm.a_v = a_v_norm
                    tree_ws_norm.c_v = c_v_norm
                    tree_ws_norm.class_nodes = tree_ws.class_nodes"""

                    warm_start_dict = {'class': tree_ws.class_nodes, 'pruned': tree_ws.pruned_nodes,
                                       'branched': tree_ws.branch_nodes, 'results': ws_assignments, 'use': True}

                    tree = TREE(h=h)
                    mbdt = MBDT_one_step(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                         time_limit=time_limit, warmstart=warm_start_dict,
                                         modelextras=model_extras, log=log, log_to_console=1)
                    # Add connectivity constraints according to model type and solve
                    mbdt.formulation()
                    mbdt.warm_start()
                    mbdt.model.update()
                    if log_files:
                        mbdt.model.write(log + '.lp')
                    mbdt.optimization()
                    if mbdt.model.RunTime < time_limit:
                        print(f'Optimal solution found in {round(mbdt.model.RunTime,4)}s. '
                              f'('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    else:
                        print('Time limit reached. ('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    mbdt.assign_tree()
                    # Uncomment to print model results
                    # UTILS.model_results(opt_model, tree)
                    print(tree.branch_nodes)
                    print(tree.class_nodes)
                    test_acc, test_assignments = UTILS.data_predict(tree=tree, data=test_set, target=mbdt.target)
                    train_acc, train_assignments = UTILS.data_predict(tree=tree, data=model_set, target=mbdt.target)
                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file, tree.height, len(mbdt.datapoints),
                             test_acc / len(test_set), train_acc / len(mbdt.datapoints), mbdt.model.Runtime,
                             mbdt.modeltype, True, mbdt_ws.model.RunTime, mbdt.time_limit, i,
                             mbdt.model.MIPGap, mbdt.model.ObjVal, mbdt.model.ObjBound,
                             mbdt.model._visnum, mbdt.model._viscuts, mbdt.model._vistime,
                             mbdt.HP_time,
                             mbdt.model._septime, mbdt.model._sepnum, mbdt.model._sepcuts,
                             mbdt.model._eps, mbdt.b_type])
                        results.close()
