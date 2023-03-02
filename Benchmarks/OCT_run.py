import sys, time, os, getopt, csv
import pandas as pd
from sklearn.model_selection import train_test_split
from OCT.OCT import OCT
from OCT.OCTH import OCTH
import UTILS as OU

# I'm so fkn sick n tired of the warnings -Kendrick Lamar Duckworth
import warnings
warnings.filterwarnings("ignore")


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
        opts, args = getopt.getopt(argv, "d:h:t:m:r:f:l:",
                                   ["data_files=", "heights=", "timelimit=",
                                    "models=", "rand_states=",
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
        elif opt in ("-f", "--results_file"):
            file_out = arg
        elif opt in ("-l", "--log_files"):
            log_files = arg

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|',
                       'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound', 'Model', 'Warm_Start', 'Warm_Start_Time',
                       'Num_CB', 'User_Cuts', 'Cuts_per_CB', 'Total_CB_Time', 'INT_CB_Time', 'FRAC_CB_Time', 'CB_Eps',
                       'Time_Limit', 'Rand_State', 'Calibration', 'Single_Feature_Use', 'Max_Features']
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

    # Using logger we log the output of the console in a text file
    # sys.stdout = OU.logger(output_path + output_name + '.txt')

    ''' We assume the target column of dataset is labeled 'target'
    Change value at your discretion '''
    target = 'target'
    numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                                                                                'glass', 'image_segmentation',
                          'ionosphere', 'parkinsons', 'iris']
    categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
    for file in data_files:
        if file in numerical_datasets: binarization = 'all-candidates'
        else: binarization = False
        # pull dataset to train model with
        data = OU.get_data(file.replace('.csv', ''), binarization=binarization)
        for h in heights:
            for i in rand_states:
                print('\nDataset: '+str(file)+', H: '+str(h)+', ' 'Rand State: '+str(i)
                      + '. Run Start: '+str(time.strftime("%I:%M %p", time.localtime())))
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                X_train, y_train = model_set.drop('target'), model_set['target']
                X_test, y_test = test_set.drop('target'), test_set['target']
                for modeltype in modeltypes:
                    method = modeltype[5:]
                    # Log .lp and .txt files name
                    if log_files:
                        log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) \
                              + '_' + 'T:' + str(time_limit)
                    else:
                        log = False
                    if method == "Univariate":
                        oct = OCT(max_depth=h, ccp_alpha=0, time_limit=time_limit, log_to_console=False)
                    elif method == "Multivariate":
                        oct = OCTH(max_depth=h, ccp_alpha=0, time_limit=time_limit, log_to_console=False)
                    oct.fit(X_train, y_train)
                    if oct.branch_rules_ is not None:
                        train_acc = oct.score(X_train, y_train)
                        test_acc = oct.score(X_test, y_test)
                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file.replace('.csv', ''), h, len(model_set), test_acc, train_acc, oct.model_.RunTime,
                             oct.model_.MIPGap, oct.model_.ObjBound, oct.model_.ObjVal, modeltype, False, 0,
                             0, 0, 0, 0, 0, 0, 0,
                             time_limit, i, False, False, False])
                        results.close()
                    if log_files:
                        oct.model_.write(log + '.lp')

