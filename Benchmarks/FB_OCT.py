from Strong_Tree.FlowOCTTree import Tree as FB_OCT_Tree
from Strong_Tree.FlowOCT import FlowOCT
from Strong_Tree.BendersOCT import BendersOCT
import Strong_Tree.FBOCTutils as FBOCTutils
from sklearn.model_selection import train_test_split
import os, time, getopt, sys, csv
import pandas as pd
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
    model_extras = None
    rand_states = None
    file_out = None
    log_files = None

    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:r:e:c:f:l:",
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
    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time',
                       'MIP_Gap', 'Obj_Val', 'Obj_Bound',
                       'Model', 'Warm_Start', 'Warm_Start_Time', 'Time_Limit', 'Rand_State']
    output_path = os.getcwd() + '/results_files/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + '_H:' + str(heights) + '_' + str(modeltypes) + \
                      '_T:' + str(time_limit) + '_' + str(model_extras) + '.csv'
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
                train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
                cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
                model_set = pd.concat([train_set, cal_set])
                OCT_tree = FB_OCT_Tree(d=h)
                # Log files
                for modeltype in modeltypes:
                    print('\nModel: ' + str(modeltype) + 'Dataset: ' + str(file) + ', H: ' + str(h) + ', Rand State: '
                          + str(i) + '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    # Log .lp and .txt files name
                    log = log_path + '_' + str(file) + '_H:' + str(h) + '_M:' + str(modeltype) \
                          + '_' + 'T:' + str(time_limit) if log_files else False

                    if 'Flow' in modeltype:
                        stoct = FlowOCT(data=model_set, label=target, tree=OCT_tree,
                                        _lambda=0, time_limit=time_limit, mode='classification')
                        stoct.create_primal_problem()
                        stoct.model.update()
                        stoct.model.optimize()
                    elif 'Benders' in modeltype:
                        stoct = BendersOCT(data=model_set, label=target, tree=OCT_tree,
                                           _lambda=0, time_limit=time_limit, mode='classification')
                        stoct.create_master_problem()
                        stoct.model.update()
                        stoct.model.optimize(FBOCTutils.mycallback)
                    if stoct.model.RunTime < time_limit:
                        print(f'Optimal solution found in {round(stoct.model.RunTime,4)}s. '
                              f'('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    else:
                        print('Time limit reached. ('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    b_value = stoct.model.getAttr("X", stoct.b)
                    beta_value = stoct.model.getAttr("X", stoct.beta)
                    p_value = stoct.model.getAttr("X", stoct.p)
                    train_acc = FBOCTutils.get_acc(stoct, train_set, b_value, beta_value, p_value)
                    test_acc = FBOCTutils.get_acc(stoct, test_set, b_value, beta_value, p_value)

                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file.replace('.csv', ''), h, len(model_set), test_acc, train_acc, stoct.model.Runtime,
                             stoct.model.MIPGap, stoct.model.ObjBound, stoct.model.ObjVal,
                             modeltype, 'N/A', 0, time_limit, i])
                        results.close()
                    if log_files:
                        stoct.model.write(log + '.lp')
