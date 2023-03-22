import MBDT_runs, warm_start_runs
import UTILS
from Benchmarks import FB_OCT, DL8_5, OCT_run, SOCT_run


numerical = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red',
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']


test_num = ['wine_white', 'iris', 'banknote', 'glass', 'climate', 'image_segmentation', 'blood', 'ionosphere']
test_cat = ['fico_binary', 'soybean_small', 'car', 'monk1', 'balance_scale', 'breast-cancer', 'monk2']
time_limit = 600
# rand_states = [138, 15, 89, 42, 0]
rand_states = [138]
file = 'test_dump.csv'
log_file = False
test_data = ['iris']
heights = [5]

############ SOCT ###############
models = ['SOCT-Benders']
warm_start = [None, 'STUMP', 'SVM']  # CHOOSE ONE
SOCT_run.main(
    ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models,
     "-r", rand_states, "-f", file, "-w", warm_start[0], "-l", log_file])


############ MBDT 2-STEP ###############
models = ['CUT1-UF']
b_type = '2-Step'  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type,
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])


"""############ MBDT ISING ###############
models = ['CUT1-UF-split', 'CUT1-UF-abs']
b_type = 'ISING'
MBDT_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type,
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])
"""

models = ['CUT1-UF-split', 'CUT1-UF-abs']
warm_start_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models,
    "-r", rand_states, "-f", file, "-e", None, "-l", log_file])
