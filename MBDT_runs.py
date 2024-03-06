#!/usr/bin/python
import os, time, getopt, sys, csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from MBDT import MBDT
from TREE import TREE
import UTILS, CALLBACKS


def main(argv):
    # print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    rand_states = None
    tuning = None
    file_out = None
    log_files = None
    weight = 0

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:c:w:l:",
                                   ["data_files=", "heights=", "time_limit=", "models=", "rand_states=",
                                    "results_file=", "calibration=", "weight=", "log_files="])
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
        elif opt in ("-c", "--tuning"):
            tuning = arg
        elif opt in ("-w", "--warm_start"):
            warmstart = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time', 'MIP_Gap', 'Obj_Val', 'Obj_Bound',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State',
                       'VIS_Calls', 'VIS_Cuts', 'VIS_Time', 'HP_Time', 'FP_Time', 'FP_Num_CB', 'FP_Num_Cuts',
                       'Eps', 'Branch_Type']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '.csv'
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
                    print('\n'+str(modeltype) +
                          ', Dataset: ' + str(file) + ', H: ' + str(h) + ', ''Rand State: ' + str(i) +
                          '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    if log_files: log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(
                        modeltype)+'_T:' + str(time_limit) + '_Seed:' + str(i)
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
                        best_cal_tree, best_cal_acc = {}, 0
                        lambda_WSV = None
                        for cal_lambda in np.linspace(0, .9, 10):
                            # Calibrate model with number of Num-Tree-size features = k for k in [1, B]
                            cal_tree = TREE(h=h)
                            cal_model = MBDT(data=cal_set, tree=cal_tree, target=target, modeltype=modeltype,
                                             time_limit=time_limit/3, warmstart=lambda_WSV, log=log, weight=cal_lambda)
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
                            cal_model.assign_tree()
                            test_cal_acc, cal_assign = UTILS.data_predict(tree=cal_tree, data=model_set,
                                                                          target=cal_model.target)
                            lambda_WSV = {'tree': cal_tree, 'data': cal_assign,
                                          'time': cal_model.model.RunTime, 'best': False}
                            if test_cal_acc > best_cal_acc:
                                weight, best_acc, best_cal_tree = cal_lambda, test_cal_acc, cal_tree
                        cal_time = time.perf_counter()-wsm_time_start
                        best_cal_acc, best_cal_assgn = UTILS.data_predict(tree=best_cal_tree, target=target, data=model_set)
                        WSV = {'tree': best_cal_tree, 'data': best_cal_assgn, 'time': cal_time, 'best': True, 'use': True}
                    tree = TREE(h=h)
                    # Model with 75% training set and time limit
                    # Specify model datapoint branching type
                    mbdt = MBDT(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                time_limit=time_limit, warmstart=WSV, log=log, weight=weight)
                    # Add connectivity constraints according to model type and solve
                    mbdt.formulation()
                    # Update with warm start values if applicable
                    if WSV is not None: mbdt.warm_start()
                    # if model_extras is not None: mbdt.extras()
                    mbdt.model.update()
                    if log_files: mbdt.model.write(log + '.lp')
                    if 'GRB' or 'ALL' in cb_type:
                        mbdt.model.optimize()
                    if 'FRAC' in cb_type:
                        # User cb.Cut FRAC S-Q cuts
                        mbdt.model.Params.PreCrush = 1
                        if '1' in cb_type: mbdt.model.optimize(CALLBACKS.frac1)
                        if '2' in cb_type: mbdt.model.optimize(CALLBACKS.frac2)
                        if '3' in cb_type: mbdt.model.optimize(CALLBACKS.frac3)
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
