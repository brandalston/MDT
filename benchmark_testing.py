from SOCTcode.SOCT.LinearClassifierHeuristic import LinearClassifierHeuristic
from SOCTcode.SOCT.SOCTStumpHeuristic import SOCTStumpHeuristic
from SOCTcode.SOCT.SOCTFull import SOCTFull
from sklearn.model_selection import train_test_split
from SOCTcode.SOCT.SOCTBenders import SOCTBenders
import pandas as pd
import csv
import time
import UTILS

target = 'target'
files = ['monk1']
rand_states = [138,15,89,42,0]
heights = [2,3,4,5]
results_file = 'benchmark_testing.csv'

for file in files:
    for h in heights:
        for rand in rand_states:
            data = UTILS.get_data(file, target)
            train_set, test_set = train_test_split(data, train_size=0.75, random_state=rand)
            X_train, X_test = train_set.loc[:, data.columns != target], test_set.loc[:, data.columns != target]
            y_train, y_test = train_set[target], test_set[target]
            print('\n\nDataset: ' + str(file) + ', H: ' + str(h) + ', Rand State: ' + str(138) +
                  '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
            soct = SOCTBenders(max_depth=h, ccp_alpha=0, time_limit=600, log_to_console=False)
            soct.fit(X_train, y_train)
            if soct.branch_rules_ is not None:
                train_acc = soct.score(X_train, y_train)
                test_acc = soct.score(X_test, y_test)
                # Update .csv file with modeltype metrics
                with open(results_file, mode='a') as results:
                    results_writer = csv.writer(results, delimiter=',', quotechar='"')
                    results_writer.writerow(
                        [file, h, len(train_set), len(train_set.columns), test_acc / len(test_set), train_acc / len(train_set),
                         soct.master_.Runtime, soct.master_.MIPGap, soct.master_.ObjVal, soct.master_.ObjBound,
                         'SOCT-Benders', soct._hp_time, 'L2-SVM', 'Full',
                         'N/A', 'N/A', 'N/A', 'N/A',
                         soct.master_._callback_time, soct.master_._callback_calls, soct.master_._callback_cuts,
                         'N/A', 600, rand, 'False', 'N/A', 'N/A'])
                    results.close()
