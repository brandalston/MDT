import MBDT_runs

"""
numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                      'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                        'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
rand_states = [138, 15, 89, 42, 0]
time_limit = 600
heights = [2, 3, 4, 5]
model_extras = None
warm_start = {'use': False, 'values': None}
file = 'test_dump_svm.csv'
b_type = 'VIS'
models = ['CUT1', 'CUT2']
gen = ((obj, rank) for obj in ['linear','quadratic'] for rank in ['|F|-1', 0.9, 0.75, 0.5, 0.25, 0.1])
data_names = ['soybean-small']
for obj, rank in gen:
    hp_info = {'objective': obj, 'rank': rank}
    test_runs.main(
        ["-d", data_names, "-h", heights, "-m", models, "-b", b_type, "-t", time_limit, "-p", hp_info,
         "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
"""

data_names = ['iris']
time_limit = 600
rand_states = [138]
file = 'ising_testing.csv'
heights = [5]
models = ['CUT1']
b_type = 'VIS'
model_extras = None
warm_start = {'use': False, 'values': None}
hp_info = None
test_runs.main(
    ["-d", data_names, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type, "-p", hp_info,
     "-r", rand_states, "-f", file, "-w", warm_start, "-e", model_extras, "-l", False])
