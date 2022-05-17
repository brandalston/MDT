#!/usr/bin/python
import os
import time
import pandas as pd
import getopt
import sys
import csv
from sklearn.model_selection import train_test_split
from sklearn import tree as HEURTree
import matplotlib.pyplot as plt
from MBDT import MBDT
from TREE import TREE
import UTILS
import RESULTS


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    rand_states = None
    warm_start = None
    file_out = None
    consol_log = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:w:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=", "warm_start=",
                                    "results_file=", "consol_log"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-m", "--model"):
            modeltypes = arg
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-w", "--tuning"):
            warm_start = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--consol_log"):
            consol_log = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model',
                       'Sep_Time', 'Sep_CB', 'Sep_Cuts', 'Sep_Avg',
                       'VIS_Time', 'VIS_CB', 'VIS_Cuts',
                       'Eps', 'Time_Limit', 'Rand_State', 'Warm Start']
    output_path = os.getcwd() + '/results_files/'

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

    # Using logger we log the output of the console in a text file
    # sys.stdout = OU.logger(output_path + output_name + '.txt')

    """ We assume the target column of dataset is labeled 'target'
    Change value at your discretion """
    target = 'target'
    for file in data_files:
        # pull dataset to train model with
        data = UTILS.get_data(file.replace('.csv', ''), target)
        for h in heights:
            for i in rand_states:
                print('\nDataset: '+str(file)+', H: '+str(h)+', '
                      'Rand State: '+str(i)+'. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                for modeltype in modeltypes:
                    if any([char.isdigit() for char in modeltype]):
                        # Generate tree and necessary structure information
                        tree = TREE(h=h)
                        # Model with 75% training set and time limit
                        opt_model = MBDT(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                         time_limit=time_limit, warm_start=warm_start, name=file)
                        # Add connectivity constraints according to model type and solve
                        opt_model.formulation()
                        opt_model.model.update()
                        opt_model.optimization()
                        opt_model.assign_tree()
                        RESULTS.model_summary(opt_model=opt_model, tree=tree, test_set=test_set,
                                              rand_state=i, results_file=out_file)
                        if consol_log:
                            consol_log_file = output_path + '_' + str(file) + '_' + str(h) + '_' + str(
                                modeltype) + '_' + 'T:' + str(time_limit) + '.txt'
                            sys.stdout = UTILS.consol_log(consol_log_file)

