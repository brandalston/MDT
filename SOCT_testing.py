from SOCTcode.SOCT.LinearClassifierHeuristic import LinearClassifierHeuristic
from SOCTcode.SOCT.SOCTStumpHeuristic import SOCTStumpHeuristic
from SOCTcode.SOCT.SOCTFull import SOCTFull
from sklearn.model_selection import train_test_split
from SOCTcode.SOCT.SOCTBenders import SOCTBenders
import pandas as pd
import time
import UTILS

file = 'monk1'
target = 'target'
data = UTILS.get_data(file, target)
train_set, test_set = train_test_split(data, train_size=0.75, random_state=138)
X_train = train_set.loc[:, data.columns != target]
y_train = train_set[target]
heights = [3,4,5]
for h in heights:
    print('\n\nDataset: ' + str(file) + ', H: ' + str(h) + ', '
                                                         'Rand State: ' + str(138) + '. Run Start: ' + str(
        time.strftime("%I:%M %p", time.localtime())))
    soct = SOCTBenders(max_depth=h, ccp_alpha=0, time_limit=600, log_to_console=False)
    soct.fit(X_train, y_train)


"""
# Use S-OCT stump warm start
start_time = time.time()
stump = SOCTStumpHeuristic(max_depth=max_depth, time_limit=600)
stump.fit(X_train, y_train)
stump_warm_start = stump.branch_rules_, stump.classification_rules_
warm_start_time = time.time() - start_time
"""