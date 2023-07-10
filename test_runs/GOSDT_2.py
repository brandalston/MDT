from Benchmarks import GOSDTg_run

numerical_datasets = ['iris', 'glass', 'blood',  'banknote', 'parkinsons',
                      'climate', 'ionosphere', 'wine_red', 'image', 'wine_white']
categorical_datasets = ['soybean_small', 'monk3', 'monk1', 'hayes_roth', 'monk2', 'house_votes_84', 'spect',
                        'breast_cancer', 'balance_scale', 'tic_tac_toe', 'car', 'kr_vs_kp', 'fico_binary']

set_1 = ['wine_white', 'iris', 'soybean_small', 'ionosphere', 'blood', 'fico_binary', 'parkinsons', 'breast-cancer']
set_2 = ['image', 'glass', 'monk1', 'wine_red', 'banknote', 'tic_tac_toe', 'climate', 'spect']

heights = [2, 3, 4, 5]
time_limit = 900
rand_states = [138, 15, 89, 42, 0]
file = 'benchmark_runs.csv'

############ GOSDT+g ###############
GOSDTg.main(["-d", set_2, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])
