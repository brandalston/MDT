import model_runs
import UTILS


time_limit = 600
rand_states = [138]
file = 'testing.csv'
heights = [4]
data_names = ['monk1']
models = ['CUT1']
# model_extras = ['regularization-3']
model_extras = None
warm_start = {'use': False, 'values': None}
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t",time_limit,
     "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
