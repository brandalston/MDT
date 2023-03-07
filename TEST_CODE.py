import MBDT_runs
import UTILS

numerical = ['iris', 'banknote', 'blood', 'climate', 'wine_white', 'wine_red',
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical = ['balance_scale', 'car', 'kr_vs_kp', 'house_votes_84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']


test_num = ['wine_white', 'iris', 'banknote', 'glass', 'climate', 'image_segmentation', 'blood', 'ionosphere']
test_cat = ['fico_binary', 'soybean_small', 'car', 'monk1', 'balance_scale', 'breast-cancer', 'monk2']
time_limit = 600
# rand_states = [138, 15, 89, 42, 0]
rand_states = [138]
file = 'benchmark_testing.csv'
log_file = False
test_data = ['house_votes_84']
heights = [2]

############ MBDT ###############
models = ['CUT1-UF','CUT1-FF-ROOT']
b_type = ['SVM']  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type[0],
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])

for u in numerical:
   data = UTILS.get_data(u)
   print(u, data.shape)