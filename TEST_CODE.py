import model_runs
import UTILS


time_limit = 600
rand_states = [138]
rand_states = [138, 15, 89, 42, 0]
file = 'testing.csv'
heights = [2,3,4,5]
data_names = ['balance-scale', 'car', 'kr-vs-kp', 'house-votes-84',
              'tic-tac-toe', 'hayes-roth', 'soybean-small', 'breast-cancer']
data_names = ['kr-vs-kp']
models = ['CUT1']
# model_extras = ['regularization-3']
model_extras = None
warm_start = {'use': False, 'values': None}
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t",time_limit,
     "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
