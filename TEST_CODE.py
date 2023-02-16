import model_runs, UTILS
"""
rand_states = [138, 15, 89, 42, 0]
rand_states = [138, 15]
time_limit = 600
heights = [2,3,4,5]
file = 'hp_variants.csv'
models = ['CUT1']
model_extras = None
warm_start = {'use': False, 'values': None}
gen = ((obj, rank) for obj in ['linear','quadratic'] for rank in ['|F|-1',0.9,0.75,0.5,0.25,0.1])
data_names = ['soybean-small']
for obj, rank in gen:
    hp_info = {'objective': obj, 'rank': rank}
    model_runs.main(
        ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-p", hp_info,
         "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
"""

"""
time_limit = 600
rand_states = [138]
file = 'test_dump_svm.csv'
heights = [2,3,4,5]
data_names = ['climate']
models = ['CUT1-SVM-UF']
# model_extras = ['regularization-3']
model_extras = None
warm_start = {'use': False, 'values': None}
hp_info = {'objective': 'quadratic', 'rank': .5}
obj_func = None

# model_runs.main(["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-p", hp_info, "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])

model_runs.main_svm(
        ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-o", obj_func,
         "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])


gen = ((obj, rank) for obj in ['linear','quadratic'] for rank in ['|F|-1',0.9,0.75])

for obj, rank in gen:
    hp_info = {'objective': obj, 'rank': rank}
    model_runs.main(
        ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-p", hp_info,
         "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
"""
bin_mode = 'all-candidates'
# bin_mode = None
numerical_datasets = ['banknote_authentication', 'blood_transfusion', 'climate_model_crashes', 'wine_white',
                      'glass_identification', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris', 'wine_red']
categorical_datasets = ['balance_scale', 'car_evaluation', 'chess', 'congressional_voting_records', 'hayes_roth',
                            'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tictactoe_endgame', 'breast_cancer',
                            'fico_binary']
for dataset in numerical_datasets:
    print('\npulling', dataset)
    data = UTILS.get_data(dataset, binarization=bin_mode)
    print(data.head(3))
    # data.to_csv(f'Datasets/{dataset}_enc.csv', header=True, index=False)