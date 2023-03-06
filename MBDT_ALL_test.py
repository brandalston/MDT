import MBDT_runs
from Benchmarks import FB_OCT, DL8_5, OCT_run, SOCT_run, GOSDTg

numerical = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons']
numerical = ['iris', 'climate', 'blood', 'ionosphere', 'glass', 'image', 'wine-white']
categorical = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'paper_runs.csv'
log_file = False

############ MBDT ###############
models = ['CUT1-ALL-ROOT', 'CUT2-ALL-ROOT']
b_type = ['SVM', 'ISING']  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type[0],
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])
