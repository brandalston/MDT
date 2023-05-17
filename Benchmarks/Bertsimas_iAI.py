import time, sys, csv, getopt, os
import pandas as pd
from sklearn.model_selection import train_test_split
import UTILS as OU
from interpretableai import iai

# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")

############ Bertsimas iAI ###############
""""
models = ['iAI-Univariate', 'iAI-Multivariate']
Bertsimas_iAI.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-r", rand_states, "-f", file])
   """

def main(argv):
    print(argv)
    data_files = None
    heights = None
    time_limit = None
    modeltypes = None
    rand_states = None
    file_out = None
    log_files = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=",
                                    "results_file="])
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
        elif opt in ("-f", "--results_file"):
            file_out = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State']
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

    ''' We assume the target column of dataname is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                          'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        data = OU.get_data(file.replace('.csv', ''))
        for h in heights:
            for i in rand_states:
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                X_train, y_train = model_set.drop('target', axis=1), model_set['target']
                X_test, y_test = test_set.drop('target', axis=1), test_set['target']

                for modeltype in modeltypes:
                    print('\n' + str(modeltype) + ', Dataset: ' + str(file) + ', H: ' + str(h) + ', Rand State: '
                          + str(i) + '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    method = modeltype[4:]
                    if method == "Univariate":
                        grid = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=h, random_seed=1))
                    elif method == "Multivariate":
                        grid = iai.GridSearch(iai.OptimalTreeClassifier(max_depth=h, random_seed=1,
                                                                        hyperplane_config={'sparsity': 'all'}))
                    start_time = time.perf_counter()
                    grid.fit(X_train, y_train)
                    run_time = time.perf_counter() - start_time
                    if run_time < time_limit:
                        print(f'Optimal solution found in {round(run_time,4)}s. '
                              f'('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    else:
                        print('Time limit reached. ('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    train_acc = grid.score(X_train, y_train, criterion='misclassification')
                    test_acc = grid.score(X_test, y_test, criterion='misclassification')

                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file.replace('.csv', ''), h, len(model_set), test_acc, train_acc, run_time,
                             'N/A', 'N/A', 'N/A', modeltype, False, 0, time_limit, i])
                        results.close()
