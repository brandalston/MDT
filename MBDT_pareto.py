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
    height = None
    time_limit = None
    modeltypes = None
    rand_states = None
    file_out = None
    b_type = None
    try:
        opts, args = getopt.getopt(argv, "d:h:t:m:b:r:f:c:",
                                   ["data_files=", "height=", "time_limit=", "models=", "branch_type=",
                                    "rand_states=", "results_file="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--data_files"):
            data_files = arg
        elif opt in ("-h", "--height"):
            height = arg
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

    ''' Columns of the results file generated '''
    summary_columns = ['Data', 'H', '|I|', 'Out_Acc', 'In_Acc', 'Sol_Time', 'MIP_Gap', 'Obj_Val'#, 'Obj_Bound',
                       'Model', 'Time_Limit', 'Rand_State']#'Warm_Start', 'Warm_Start_Time',
                       #'VIS_Calls', 'VIS_Cuts', 'VIS_Time', 'HP_Time', 'FP_Time', 'FP_Num_CB', 'FP_Num_Cuts',
                       #'Eps', 'Branch_Type']
    output_path = os.getcwd() + '/'
    log_path = os.getcwd() + '/log_files/'
    if file_out is None:
        output_name = str(data_files) + 'Pareto' + '_H:' +str(height) + '_' + str(b_type) + '_' + str(modeltypes) + \
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
        data = UTILS.get_data(dataname=file.replace('.csv', ''), binarization=None)
        for i in rand_states:
            train_set, test_set = train_test_split(data, train_size=0.5, random_state=i)
            cal_set, test_set = train_test_split(test_set, train_size=0.5, random_state=i)
            model_set = pd.concat([train_set, cal_set])
            for modeltype in modeltypes:
                WSV = {'use': False}
                for num_features in range(1, 2 ** int(height)):
                    print('\n' + str(modeltype) + '-' + str(b_type) +
                          ', Dataset: ' + str(file) + ', |F|: ' + str(num_features) + ', ''Rand State: ' + str(i) +
                          '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
                    extras = [f'num_branch-{num_features}']
                    tree = TREE(h=height)
                    if b_type == 'one-step':
                        mbdt = MBDT_one_step(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                             time_limit=time_limit, warmstart=WSV,
                                             modelextras=extras, log_to_console=0)
                    else:
                        mbdt = MBDT(data=model_set, tree=tree, target=target, modeltype=modeltype,
                                    time_limit=time_limit, warmstart=WSV,
                                    modelextras=extras, log_to_console=0)
                    # Add connectivity constraints according to model type and solve
                    mbdt.formulation()
                    mbdt.extras()
                    if WSV['use']: mbdt.warm_start()
                    mbdt.model.update()
                    mbdt.optimization()
                    if mbdt.model.RunTime < time_limit:
                        print(f'Optimal solution found in {round(mbdt.model.RunTime,4)}s. '
                              f'('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    else:
                        print('Time limit reached. ('+str(time.strftime("%I:%M %p", time.localtime()))+')')
                    mbdt.assign_tree()
                    test_acc, test_assignments = UTILS.data_predict(tree=tree, data=test_set, target=mbdt.target)
                    train_acc, train_assignments = UTILS.data_predict(tree=tree, data=model_set, target=mbdt.target)

                    ws_acc, ws_assignments = UTILS.data_predict(tree=tree, data=model_set, target=mbdt.target)
                    WSV = {'class': tree.class_nodes, 'pruned': tree.pruned_nodes, 'branched': tree.branch_nodes,
                           'results': ws_assignments, 'use': True}

                    with open(out_file, mode='a') as results:
                        results_writer = csv.writer(results, delimiter=',', quotechar='"')
                        results_writer.writerow(
                            [file, tree.height, len(mbdt.datapoints),
                             test_acc / len(test_set), train_acc / len(mbdt.datapoints), mbdt.model.Runtime,
                             mbdt.model.MIPGap, mbdt.model.ObjVal, mbdt.model.ObjBound,
                             mbdt.modeltype, WSV['use'], 0, mbdt.time_limit, i,
                             mbdt.model._visnum, mbdt.model._viscuts, mbdt.model._vistime,
                             mbdt.HP_time, mbdt.svm_branches, len(tree.branch_nodes),
                             mbdt.model._septime, mbdt.model._sepnum, mbdt.model._sepcuts,
                             mbdt.model._eps, mbdt.b_type])
                        results.close()
                    del tree, mbdt
        pareto_data = pd.read_csv(os.getcwd() + '/results_files/'+file_out, na_values='?')
        for file in ['monk1', 'soy', 'ion', 'iris']:
            file_data = pareto_data[pareto_data['Data'] == file.replace('.csv', '')]
            frontier_avg = pd.DataFrame(columns=summary_columns)
            # print(file_data.head(5))
            for model in ['CUT_w-H', 'CUT-H']:
                sub_data = file_data.loc[file_data['Model'] == model]
                for feature in sub_data['Branch_Num'].unique():
                    subsub_data = sub_data.loc[sub_data['Branch_Num'] == feature]
                    frontier_avg = frontier_avg.append({
                        'Data': file.replace('.csv', ''), 'H': int(subsub_data['H'].mean()),
                        '|I|': int(subsub_data['|I|'].mean()),
                        'Out_Acc': 100 * subsub_data['Out_Acc'].mean(), 'In_Acc': 100 * subsub_data['In_Acc'].mean(),
                        'Sol_Time': subsub_data['Sol_Time'].mean(), 'MIP_Gap': 100 * subsub_data['MIP_Gap'].mean(),
                        'Obj_Val': subsub_data['Obj_Val'].mean(),
                        'Model': model,
                        #'Num_CB': subsub_data['Num_CB'].mean(), 'User_Cuts': subsub_data['User_Cuts'].mean(),
                        #'Cuts_per_CB': subsub_data['Cuts_per_CB'].mean(),
                        #'Total_CB_Time': subsub_data['Total_CB_Time'].mean(),
                        #'INT_CB_Time': subsub_data['INT_CB_Time'].mean(),
                        #'FRAC_CB_Time': subsub_data['FRAC_CB_Time'].mean(), 'CB_Eps': subsub_data['CB_Eps'].mean(),
                        'Time_Limit': time_limit, 'Rand_State': 'None','Max_Features': float(feature)
                        #'Calibration': False, 'Single_Feature_Use': False,
                    }, ignore_index=True)
            UTILS.pareto_plot(frontier_avg, types=['time','acc'])
