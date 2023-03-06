import MBDT_runs

numerical = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
categorical = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
time_limit = 600
# rand_states = [138, 15, 89, 42, 0]
rand_states = [138]
file = 'benchmark_testing.csv'
log_file = False
test_data=['iris']
heights = [2]

############ MBDT ###############
models = ['CUT1-FF-ROOT', 'CUT2','CUT1-UF','CUT2-UF']
b_type = ['SVM', 'ISING']  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type[0],
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])
