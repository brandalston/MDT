from Benchmarks import FB_OCT, DL8_5, OCT_run, SOCT_run
try:
   from Benchmarks import GOSDTg
except:
   print("gosdt modeule not found")
numerical_datasets = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                      'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
categorical_datasets = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                        'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']

test_data = ['blood', 'hayes_roth']
heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'benchmarks_testing.csv'
log_file = False

############ SOCT ###############
models = ['SOCT-Full', 'SOCT-Benders']
warm_start = ['STUMP', 'SVM']  # CHOOSE ONE
SOCT_run.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, 
    "-r", rand_states, "-f", file, "-w", warm_start[0], "-l", log_file])

############ OCT ###############
models = ['OCT-Univariate', 'OCT-Multivariate']
print(models[0][4:])
OCT_run.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-r", rand_states, "-f", file, "-l", log_file])

############ DL8.5 ###############
DL8_5.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])

############ GOSDT+g ###############
GOSDTg.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])

############ Bertsimas iAI ###############
""""
models = ['iAI-Univariate', 'iAI-Multivariate']
Bertsimas_iAI.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-r", rand_states, "-f", file])
   """
