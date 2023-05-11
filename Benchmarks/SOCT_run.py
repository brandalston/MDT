import sys, time, csv, os, getopt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import UTILS as OU
from Benchmarks.SOCT.LinearClassifierHeuristic import LinearClassifierHeuristic
from Benchmarks.SOCT.SOCTStumpHeuristic import SOCTStumpHeuristic
from Benchmarks.SOCT.SOCTFull import SOCTFull
from Benchmarks.SOCT.SOCTBenders import SOCTBenders

# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")


def main(argv):
    # print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    rand_states = None
    warm_start = None
    file_out = None
    log_files = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:w:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=", "warm_start=",
                                    "results_file=", "log_files"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--heights"):
            heights = arg
        elif opt in ("-t", "--timelimit"):
            time_limit = int(arg)
        elif opt in ("-m", "--models"):
            modeltypes = arg
        elif opt in ("-r", "--rand_states"):
            rand_states = arg
        elif opt in ("-w", "--warm_start"):
            warm_start = arg
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'VIS_calls', 'VIS_cuts', 'VIS_time', 'HP_time']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_' + str(warm_start) + '.csv'
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

    ''' We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                          'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        data = OU.get_data(file.replace('.csv', ''), binarization=None)
        for h in heights:
            for i in rand_states:
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                X_train, y_train = model_set.drop('target', axis=1), model_set['target']
                X_test, y_test = test_set.drop('target', axis=1), test_set['target']
                X_valid, y_valid = cal_set.drop('target', axis=1), cal_set['target']
                data_map = {i: X_train.index[i] for i in range(len(X_train))}
                for modeltype in modeltypes:
                    print('\n' + str(modeltype) + ', Warm Start: ' + str(warm_start)+ ', Dataset: ' + str(file) +
                          ', H: ' + str(h) + ', Rand State: ' + str(i) +
                          '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    method = modeltype[5:]
                    alphas_to_try = [0.00001, 0.0001, 0.001, 0.01, 0.1]
                    best_ccp_alpha = min(alphas_to_try)
                    warm_start_time = 0
                    if warm_start == 'SVM':
                        start_time = time.perf_counter()
                        best_valid_acc = 0
                        for ccp_alpha in alphas_to_try:
                            # For the purposes of tuning alpha, use an SVM warm start
                            lch = LinearClassifierHeuristic(max_depth=h,
                                                            linear_classifier=LinearSVC(random_state=0))
                            lch.fit(X_valid, y_valid)
                            test_svm_warm_start = lch.branch_rules_, lch.classification_rules_
                            if method == "Full":
                                soct = SOCTFull(max_depth=h, ccp_alpha=ccp_alpha, warm_start_tree=test_svm_warm_start,
                                                time_limit=time_limit/5, log_to_console=False)
                            elif method == "Benders":
                                soct = SOCTBenders(max_depth=h, ccp_alpha=ccp_alpha, warm_start_tree=test_svm_warm_start,
                                                   time_limit=time_limit/5, log_to_console=True)
                            soct.fit(X_valid, y_valid)
                            if soct.branch_rules_ is not None:
                                valid_acc = soct.score(X_test, y_test)
                                if valid_acc > best_valid_acc:
                                    best_ccp_alpha = ccp_alpha
                                    warm_start = test_svm_warm_start
                                    best_valid_acc = valid_acc
                            else: print("Tuning timed out on ccp_alpha =", ccp_alpha)
                        warm_start_time = time.perf_counter()-start_time
                    elif warm_start == 'STUMP':
                        start_time = time.perf_counter()
                        stump = SOCTStumpHeuristic(max_depth=h, time_limit=time_limit/10)
                        stump.fit(X_train, y_train)
                        warm_start = stump.branch_rules_, stump.classification_rules_
                        warm_start_time = time.perf_counter() - start_time
                    if log_files:
                        log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) + \
                              '_T:' + str(time_limit) + '_W:' + str(warm_start)
                    else: log = None
                    if method == "Full":
                        soct = SOCTFull(max_depth=h, ccp_alpha=best_ccp_alpha, warm_start_tree=warm_start,
                                        time_limit=time_limit, log_to_console=False, log=log)
                    elif method == "Benders":
                        soct = SOCTBenders(max_depth=h, ccp_alpha=best_ccp_alpha, warm_start_tree=warm_start,
                                           time_limit=time_limit, log_to_console=True, log=log)
                    soct.fit(X_train, y_train)
                    if soct.model_.RunTime < time_limit:
                        print(f'Optimal solution found in {round(soct.model_.RunTime,4)}s. '
                              f'('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    else:
                        print('Time limit reached. ('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    if soct.branch_rules_ is not None:
                        train_acc = soct.score(X_train, y_train)
                        test_acc = soct.score(X_test, y_test)
                    # a_v, b_v, paths = soct.solution_values(data_map)
                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file.replace('.csv', ''), h, len(model_set), test_acc, train_acc, soct.model_.RunTime,
                             modeltype, str(warm_start), warm_start_time, time_limit, i,
                             soct.model_.MIPGap, soct.model_.ObjBound, soct.model_.ObjVal,
                             soct.model_._callback_calls, soct.model_._callback_cuts, soct.model_._callback_time, soct.hp_time])
                        results.close()
