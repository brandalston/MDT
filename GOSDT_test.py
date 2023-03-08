import MBDT_runs
from Benchmarks import FB_OCT, DL8_5, OCT_run, SOCT_run, GOSDTg

numerical = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons']
categorical = ['house_votes_84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
test_set = ['wine_white', 'iris', 'banknote', 'glass', 'climate', 'image_segmentation', 'blood', 'ionosphere',
            'fico_binary', 'soybean_small', 'car', 'monk1', 'balance_scale', 'breast-cancer', 'monk2']
finished = ['car', 'balance', 'kvk']

test_num = ['wine_white', 'iris', 'banknote', 'glass', 'climate', 'image', 'blood', 'ionosphere']
test_cat = ['fico_binary', 'soybean_small', 'car', 'monk1', 'balance_scale', 'breast-cancer', 'monk2']
heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'paper_runs_2.csv'
log_file = False

############ GOSDT+g ###############
GOSDTg.main(
   ["-d", test_num+test_cat, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])
