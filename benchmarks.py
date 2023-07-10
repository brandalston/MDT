try: from Benchmarks import GOSDTg_run
except: pass

"""
numerical = ['iris', 'banknote', 'blood', 'climate', 'wine-white', 'wine-red'
                      'glass', 'image_segmentation', 'ionosphere', 'parkinsons', 'iris']
categorical = ['balance_scale', 'car', 'kr_vs_kp', 'house-votes-84', 'hayes_roth', 'breast_cancer',
                        'monk1', 'monk2', 'monk3', 'soybean_small', 'spect', 'tic_tac_toe', 'fico_binary']
heights = [2, 3, 4, 5]
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
file = 'paper_runs.csv'
log_file = False

############ MBDT ###############
models = ['CUT1', 'MBDT']
b_type = ['SVM', 'ISING']  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type[0],
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])


############ SOCT ###############
models = ['SOCT-Benders', 'SOCT-Full']
warm_start = [None,'STUMP', 'SVM']  # CHOOSE ONE
SOCT_run.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-m", models, 
    "-r", rand_states, "-f", file, "-w", warm_start[0], "-l", log_file])

############ OCT ###############
models = ['OCT-Univariate', 'OCT-Multivariate']
OCT_run.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-m", models,
    "-r", rand_states, "-f", file, "-l", log_file])

############ DL8.5 ###############
DL8_5.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])

############ GOSDT+g ###############
GOSDTg.main(
   ["-d", categorical+numerical, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])
"""

test_data = ['soybean_small']
heights = [2]
time_limit = 600
file = 'benchmark_testing.csv'
log_file = False
rand_states = [138]

"""############ SOCT ###############
models = ['SOCT-Benders']
warm_start = [None, 'STUMP', 'SVM']  # CHOOSE ONE
SOCT_run.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models,
    "-r", rand_states, "-f", file, "-w", warm_start[0], "-l", log_file])

############ MBDT ###############
models = ['CUT1']
b_type = ['SVM', 'ISING']  # CHOOSE ONE
extras = None
warm_start = {'use': False, 'values': None}
MBDT_runs.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-b", b_type[0],
    "-r", rand_states, "-f", file, "-e", extras, "-w", warm_start, "-l", log_file])

############ OCT ###############
models = ['OCT-Multivariate']
OCT_run.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-m", models, "-r", rand_states, "-f", file, "-l", log_file])

############ DL8.5 ###############
DL8_5.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])
"""
############ GOSDT+g ###############
GOSDTg.main(
   ["-d", test_data, "-h", heights, "-t", time_limit, "-r", rand_states, "-f", file])
