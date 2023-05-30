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
heights = [3]

"""############ SOCT ###############
models = ['SOCT-Benders']
# warm_start = [None, 'STUMP', 'SVM']  # CHOOSE ONE
SOCT_run.main(
    ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models,
     "-r", rand_states, "-f", file, "-w", None, "-l", log_file])"""

"""############ MBDT 2-STEP ###############
models = ['CUT1-UF']
b_type = 'two-step'  # CHOOSE ONE
MBDT_runs.main(
    ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type, "-c", 0,
     "-r", rand_states, "-f", file, "-e", None, "-w", {'use': False}, "-l", log_file])

############ MBDT ONE STEP ###############
models = ['CUT1-UF-trad', 'CUT1-trad-2'] # , 'CUT1-UF-abs', 'CUT1-UF-trad']
b_type = 'one-step'
MBDT_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type, "-c", 0,
    "-r", rand_states, "-f", file, "-e", None, "-w", {'use': False, 'values': None}, "-l", log_file])"""

############ MBDT ISING w/ 2-STEP WARM START ###############
models = ['CUT1-UF-trad']
warm_start_runs.main(
    ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models,
     "-r", rand_states, "-f", file, "-e", None, "-l", log_file, "-c", 0])

models = ['CUT1-UF-trad-2']
warm_start_runs.main(
    ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models,
     "-r", rand_states, "-f", file, "-e", None, "-l", log_file, "-c", 1])