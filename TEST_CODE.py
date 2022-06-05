import model_runs

# """
time_limit = 600
rand_states = [138]
# rand_states = [138, 15, 89, 42, 0]
file = 'testing.csv'
heights = [5]
data_names = ['monk1']
models = ['CUT1']
# model_extras = ['regularization-3']
model_extras = None
warm_start = {'use': False, 'values': None}
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t",time_limit,
     "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
"""
time_limit = 600
rand_states = [138, 15, 89, 42, 0]
models = ['CUT1']
heights = [5]
file = 'testing_3.csv'
data_names = ['car']
model_extras = None
warm_start = {'use': False, 'values': None}
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t",time_limit,
     "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
     
heights = [2,3,4,5]
data_names = ['kr-vs-kp', 'house-votes-84',
              'tic-tac-toe', 'hayes-roth', 'soybean-small', 'breast-cancer']
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t",time_limit,
     "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
"""