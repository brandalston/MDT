import MBDT_runs
from Benchmarks import FB_OCT, DL8_5, OCT_run, SOCT_run, GOSDTg

numerical = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical = ['house_votes_84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
test_set = ['wine_white', 'iris', 'banknote', 'glass', 'climate', 'image_segmentation', 'blood', 'ionosphere',
            'fico_binary', 'soybean_small', 'car', 'monk1', 'balance_scale', 'breast-cancer', 'monk2']

test_num = ['wine_white', 'iris', 'banknote', 'glass', 'climate', 'image', 'blood', 'ionosphere']
test_cat = ['fico_binary', 'soybean_small', 'car', 'monk1', 'balance_scale', 'breast-cancer', 'monk2']
heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'paper_runs_2.csv'
log_file = False

############ MBDT ###############
models = ['CUT1-ALL-ROOT', 'CUT2-ALL-ROOT']
b_type = ['SVM', 'ISING']  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", test_num+test_cat, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type[0],
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])
