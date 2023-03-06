import MBDT_runs
from Benchmarks import FB_OCT, DL8_5, OCT_run, SOCT_run, GOSDTg

numerical = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
             'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
categorical = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
               'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'paper_runs.csv'
log_file = False

############ GOSDT+g ###############
GOSDTg.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])
