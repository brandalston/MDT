#!/usr/bin/python
import os, time, getopt, sys, csv
import pandas as pd
import numpy as np
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
    warmstart = None
    model_extras = None
    file_out = None
    log_files = None
    b_type = None
    console_log = None
    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:b:r:f:e:w:l:c:",
                                   ["data_files=", "heights=", "time_limit=", "models=", "branch_type=", "rand_states=",
                                    "results_file=", "model_extras=", "warm_start=", "log_files=", "console_log="])
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
        elif opt in ("-b", "--branch_type"):
            b_type = arg
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-e", "--model_extras"):
            model_extras = arg
        elif opt in ("-w", "--warm_start"):
            warmstart = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg
        elif opt in ("-c", "--console_log"):
            console_log = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time', 'MIP_Gap', 'Obj_Val', 'Obj_Bound',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State',
                       'VIS_Calls', 'VIS_Cuts', 'VIS_Time', 'HP_Time', 'FP_Time', 'FP_Num_CB', 'FP_Num_Cuts',
                       'Eps', 'Branch_Type']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(b_type) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_E:' + str(model_extras) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()
    """ We assume the target column of dataname is labeled 'target'
    Change value at your discretion """
    target = 'target'

    for file in data_files:
        # if file in numerical_datasets: binarization = 'all-candidates'
        # else: binarization = False
        # pull dataname to train model with
        data = UTILS.get_data(dataname=file.replace('.csv', ''), binarization=None)
        for h in heights:
            for i in rand_states:
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                # data.dataset = model_set
                for modeltype in modeltypes:
                    print('\n'+str(modeltype)+'-'+str(b_type) +
                          ', Dataset: ' + str(file) + ', H: ' + str(h) + ', ''Rand State: ' + str(i) +
                          '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    if log_files: log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(
                        modeltype)+'-'+str(b_type)+ '_T:' + str(time_limit) + '_Seed:' + str(i) + '_E:' + str(
                        model_extras)
                    else: log = None
                    # Generate tree and necessary structure information
                    cb_type = modeltype[5:]
                    if len(cb_type) == 0: cb_type = 'ALL'
                    print('\n' + str(file) + ', H_' + str(h) + ', ' + str(modeltype) + ', Rand_' + str(i)
                          + '. Run Start: ' + str(time.strftime("%I:%M:%S %p", time.localtime())))
                    # Log .lp and .txt files name
                    WSV = None
                    if tuning:
                        wsm_time_start = time.perf_counter()
                        best_tree, best_acc = {}, 0
                        lambda_WSV = None
                        for cal_lambda in np.linspace(0, .9, 10):
                            # Calibrate model with number of Num-Tree-size features = k for k in [1, B]
                            cal_tree = TREE(h=h)
                            cal_model = MBDT(data=cal_set, tree=cal_tree, target=target, model=modeltype, name=file,
                                             time_limit=0.5*time_limit, warm_start=lambda_WSV, weight=cal_lambda)
                            cal_model.formulation()
                            if lambda_WSV is not None: cal_model.warm_start()
                            cal_model.model.update()
                            print('test:', round(cal_lambda, 2), 'start:',str(time.strftime("%I:%M:%S %p", time.localtime())))
                            if 'GRB' or 'ALL' in cb_type:
                                cal_model.model.optimize()
                            if 'FRAC' in cb_type:
                                # User cb.Cut FRAC S-Q cuts
                                cal_model.model.Params.PreCrush = 1
                                if '1' in cb_type: cal_model.model.optimize(CALLBACKS.frac1)
                                if '2' in cb_type: cal_model.model.optimize(CALLBACKS.frac2)
                                if '3' in cb_type: cal_model.model.optimize(CALLBACKS.frac3)
                            UTILS.node_assign(cal_model, cal_tree)
                            UTILS.tree_check(cal_tree)
                            cal_acc, cal_assign = UTILS.model_acc(tree=cal_tree, target=target, data=cal_set)
                            lambda_WSV = {'tree': cal_tree, 'data': cal_assign,
                                          'time': cal_model.model.RunTime, 'best': False}
                            if cal_acc > best_acc:
                                weight, best_acc, best_tree = cal_lambda, cal_acc, cal_tree
                        cal_time = time.perf_counter()-wsm_time_start
                        model_wsm_acc, model_wsm_assgn = UTILS.model_acc(tree=best_tree, target=target, data=model_set)
                        WSV = {'tree': best_tree, 'data': model_wsm_assgn, 'time': cal_time, 'best': True}
                    tree = TREE(h=h)
                    # Model with 75% training set and time limit
                    # Specify model datapoint branching type
                    if b_type == 'one-step':
                        mbdt = MBDT_one_step(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                             time_limit=time_limit, warmstart=warmstart,
                                             modelextras=model_extras, log=log, log_to_console=console_log)
                    else:
                        mbdt = MBDT(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                    time_limit=time_limit, warmstart=warmstart,
                                    modelextras=model_extras, log=log, log_to_console=console_log)
                    # Add connectivity constraints according to model type and solve
                    mbdt.formulation()
                    if warmstart['use']: mbdt.warm_start()
                    if model_extras is not None: mbdt.extras()
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
                    test_acc, test_assignments = UTILS.data_predict(tree=tree, data=test_set, target=mbdt.target)
                    train_acc, train_assignments = UTILS.data_predict(tree=tree, data=model_set, target=mbdt.target)
                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file, tree.height, len(mbdt.datapoints),
                             test_acc / len(test_set), train_acc / len(mbdt.datapoints), mbdt.model.Runtime,
                             mbdt.model.MIPGap, mbdt.model.ObjVal, mbdt.model.ObjBound,
                             mbdt.modeltype, warmstart['use'], 0, mbdt.time_limit, i,
                             mbdt.model._visnum, mbdt.model._viscuts, mbdt.model._vistime,
                             mbdt.HP_time, mbdt.svm_branches, len(tree.branch_nodes),
                             mbdt.model._septime, mbdt.model._sepnum, mbdt.model._sepcuts,
                             mbdt.model._eps, mbdt.b_type])
                        results.close()
                    del tree, mbdt
