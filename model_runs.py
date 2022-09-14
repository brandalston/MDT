#!/usr/bin/python
import os
import time
import pandas as pd
import getopt
import sys
import csv
from sklearn.model_selection import train_test_split
from MBDT import MBDT
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
    hp_info = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:w:e:f:p:l:",
                                   ["data_files=", "heights=", "time_limit=",
                                    "models=", "rand_states=", "warm_start=",
                                    "model_extras=", "results_file=", "hp_obj=", "log_files"])
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
        elif opt in ("-w", "--warm_start"):
            warmstart = arg
        elif opt in ("-e", "--model_extras"):
            model_extras = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-p", "--hp_info"):
            hp_info = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', '|F|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'HP_Time', 'HP_Size', 'HP_Obj', 'HP_Rank',
                       'FP_CB_Time', 'FP_Num_CB', 'FP_Cuts', 'FP_Avg',
                       'VIS_CB_Time', 'VIS_Num_CB', 'VIS_Cuts',
                       'Eps', 'Time_Limit', 'Rand_State',
                       'Warm Start', 'Regularization', 'Max_Features']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_E:'+ str(model_extras)+'.csv'
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

    for file in data_files:
        # pull dataset to train model with
        data = UTILS.get_data(file, target)
        for h in heights:
            for i in rand_states:
                print('Dataset: '+str(file)+', H: '+str(h)+', '
                      'Rand State: '+str(i)+'. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                model_set.name = data.name
                for modeltype in modeltypes:
                    if any([char.isdigit() for char in modeltype]):
                        # Log .lp and .txt files name
                        log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(
                                modeltype) + '_T:' + str(time_limit) + '_Seed:' + str(i) + '_E:' + str(
                                model_extras)
                        if not log_files: log = None
                        # Generate tree and necessary structure information
                        tree = TREE(h=h)
                        # Model with 75% training set and time limit
                        opt_model = MBDT(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                         time_limit=time_limit, warmstart=warmstart, hp_info=hp_info,
                                         modelextras=model_extras, log=log)
                        # Add connectivity constraints according to model type and solve
                        opt_model.formulation()
                        if warmstart['use']: opt_model.warm_start()
                        if model_extras is not None: opt_model.extras()
                        opt_model.model.update()
                        if log_files: opt_model.model.write(log+'.lp')
                        opt_model.optimization()
                        print(f'Optimal solution found in {round(opt_model.model.Runtime/60, 4)} min.'
                                  f'({time.strftime("%I:%M %p", time.localtime())})') if \
                            opt_model.model.RunTime < time_limit else \
                            print(f'Time limit reached. ({time.strftime("%I:%M %p", time.localtime())})')
                        opt_model.assign_tree()
                        # Uncomment to print model results
                        # UTILS.model_results(opt_model.model, opt_model.tree)
                        UTILS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                            rand_state=i, results_file=out_file)
