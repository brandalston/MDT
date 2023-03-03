'''
This file is the implementation of the GOSDT+guesses model found in the paper ''[Generalized and Scalable Optimal Sparse Decision Trees](http://proceedings.mlr.press/v119/lin20g/lin20g.pdf)''.
and publicly available on https://pypi.org/project/gosdt/
Code is taken directly from https://pypi.org/project/gosdt/
All rights and ownership are to the original owners.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import time, pathlib, csv, getopt, sys
from sklearn.ensemble import GradientBoostingClassifier
from gosdt.model.threshold_guess import compute_thresholds
from gosdt.model.gosdt import GOSDT
import UTILS as OU
# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")


def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    rand_states = None
    file_out = None
    try:
        opts, args = getopt.getopt(argv, "d:h:t:r:f:",
                                   ["data_files=", "heights=", "timelimit=", "rand_states=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State']
    output_path = 'results_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_GOSDT+g.5' + \
                      '_T:' + str(time_limit) + '.csv'
    else:
        output_name = file_out
    out_file = output_path + output_name
    if file_out is None:
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(summary_columns)
            f.close()

    ''' We assume the target column of dataset is labeled 'target'
        Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                                                                                'glass', 'image_segmentation',
                          'ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']

    for file in data_files:
        data = OU.get_data(file.replace('.csv', ''))
        for h in heights:
            for i in rand_states:
                # data split
                train_set, test_set = train_test_split(data, train_size=0.75, random_state=i)
                X_train, Y_train = train_set.drop('target', axis=1), train_set['target']
                X_test, Y_test = test_set.drop('target', axis=1), test_set['target']

                print('\nGOSDT+g, Dataset: ' + str(file) + ', H: ' + str(h) + ', Rand State: '
                      + str(i) + '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                # guess thresholds and lower bounds
                n_est = 40
                model_x_train, thresholds, header, threshold_guess_time = compute_thresholds(X_train, Y_train, n_est, h)
                model_y_train = pd.DataFrame(Y_train)
                # guess lower bound
                start_time = time.perf_counter()
                clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=h, random_state=i)
                clf.fit(model_x_train, model_y_train.values.flatten())
                warm_labels = clf.predict(model_x_train)
                lb_time = time.perf_counter() - start_time

                # save the labels from lower bound guesses as a tmp file and return the path to it.
                labelsdir = pathlib.Path('/tmp/warm_lb_labels')
                labelsdir.mkdir(exist_ok=True, parents=True)
                labelpath = labelsdir / 'warm_label.tmp'
                labelpath = str(labelpath)
                pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels", index=None)
                # initialize GOSDT, train and predict
                config = {
                    "regularization": 1 / len(model_x_train),
                    "depth_budget": h + 1,
                    "warm_LB": False,
                    "path_to_labels": labelpath,
                    "time_limit": time_limit,
                    "similar_support": False,
                    "look_ahead": False,
                    "worker limit": 1,
                    "feature_transform": False
                }
                start = time.perf_counter()
                model = GOSDT(config)
                model.fit(model_x_train, model_y_train)
                model_time = time.perf_counter() - start
                if model_time < time_limit:
                    print(f'Optimal solution found in {round(model, 4)}s. '
                          f'(' + str(time.strftime("%I:%M %p", time.localtime())) + ')')
                else:
                    print('Time limit reached. '+str(time.strftime("%I:%M %p", time.localtime())))
                test_acc = model.score(X_test, Y_test)
                train_acc = model.score(X_train, Y_train),
                with open(out_file, mode='a') as results:
                    results_writer = csv.writer(results, delimiter=',', quotechar='"')
                    results_writer.writerow(
                        [file.replace('.csv', ''), h, len(train_set), test_acc, train_acc, model_time,
                         'GOSDT+g', 'SVM', lb_time,  time_limit, i])
                    results.close()
