import MBDT_runs

numerical_datasets = ['iris', 'glass', 'blood',  'banknote', 'parkinsons',
                      'climate', 'ionosphere', 'wine_red', 'image', 'wine_white']
categorical_datasets = ['soybean_small', 'monk3', 'monk1', 'hayes_roth', 'monk2', 'house_votes_84', 'spect',
                        'breast_cancer', 'balance_scale', 'tic_tac_toe', 'car', 'kr_vs_kp', 'fico_binary']

set_1 = ['wine_white', 'iris', 'soybean_small', 'ionosphere', 'blood', 'fico_binary', 'parkinsons', 'breast-cancer']
set_2 = ['image', 'glass', 'monk1', 'wine_red', 'banknote', 'tic_tac_toe', 'climate', 'spect']

heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'benchmark_runs.csv'

############ MBDT ###############
models = ['CUT1', 'CUT2']
b_type = 'two-step'  # CHOOSE ONE
MBDT_runs.main(
    ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type, "-c", 0,
     "-r", rand_states, "-f", file, "-e", None, "-w", {'use': False}, "-l", False])