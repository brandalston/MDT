"""
import model_runs
rand_states = [138,15,89,42,0]
time_limit = 600
heights = [2,3,4,5]
file = 'testing_svm.csv'
models = ['CUT1']
model_extras = None
warm_start = {'use': False, 'values': None}
obs, ranks = ['linear','quadratic'], ['full','|F|-1',0.9,0.75,0.5,0.25,0.1]
gen = ((obj, rank) for obj in obs for rank in ranks)
data_names = ['monk1']
for obj, rank in gen:
    hp_info = {'objective': obj, 'rank': rank}
    print('\n\nHP TYPE:', hp_info)
    model_runs.main(
        ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-p", hp_info,
         "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
"""

from SOCTcode.SOCT.LinearClassifierHeuristic import LinearClassifierHeuristic
from SOCTcode.SOCT.SOCTStumpHeuristic import SOCTStumpHeuristic
from SOCTcode.SOCT.SOCTFull import SOCTFull
from sklearn.model_selection import train_test_split
from SOCTcode.SOCT.SOCTBenders import SOCTBenders
import pandas as pd
import csv
import os
import time
import UTILS

target = 'target'
# files = ['soybean-small','banknote_authentication','breast-cancer','hayes-roth',
#          'tic-tac-toe','house-votes-84','balance-scale','climate']
files = ['monk1']
rand_states = [138, 15, 89, 42, 0]
heights = [2,3,4,5]
time_lim = 600

output_path = os.getcwd() + '/results_files/'
out_file = output_path + 'SOCT_benchmark.csv'

for file in files:
    for h in heights:
        for rand in rand_states:
            data = UTILS.get_data(file, target)
            train_set, test_set = train_test_split(data, train_size=0.75, random_state=rand)
            X_train, X_test = train_set.loc[:, data.columns != target], test_set.loc[:, data.columns != target]
            y_train, y_test = train_set[target], test_set[target]
            print('\n\nDataset: ' + str(file) + ', H: ' + str(h) + ', Rand State: ' + str(rand) +
                  '. Run Start: ' + str(time.strftime("%I:%M %p", time.localtime())))
            soct = SOCTBenders(max_depth=h, ccp_alpha=0, time_limit=time_lim, log_to_console=False)
            soct.fit(X_train, y_train)
            if soct.branch_rules_ is not None:
                train_acc, test_acc = soct.score(X_train, y_train), soct.score(X_test, y_test)
                # Update .csv file with modeltype metrics
                with open(out_file, mode='a') as results:
                    results_writer = csv.writer(results, delimiter=',', quotechar='"')
                    results_writer.writerow(
                        [file, h, len(train_set), len(train_set.columns),
                         test_acc, train_acc, soct.master_.Runtime,
                         soct.master_.MIPGap, soct.master_.ObjVal, soct.master_.ObjBound,
                         soct._hp_time, soct._svm_hp, soct._generic_hp, soct._total_branch,
                         soct.master_._callback_time, soct.master_._callback_calls, soct.master_._callback_cuts,
                         time_lim, rand])
                    results.close()
