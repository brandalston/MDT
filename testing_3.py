import model_runs
rand_states = [138,15,89,42,0]
time_limit = 600
heights = [2,3,4,5]
file = 'hp_size_testing.csv'
models = ['CUT1']
model_extras = None
warm_start = {'use': False, 'values': None}
gen = ((obj, rank) for obj in ['linear','quadratic'] for rank in ['full','|F|-1',0.9,0.75,0.5,0.25,0.1])
data_names = ['breast-cancer']
for obj, rank in gen:
    hp_info = {'objective': obj, 'rank': rank}
    model_runs.main(
        ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-p", hp_info,
         "-r", rand_states, "-w", warm_start, "-e", model_extras, "-f", file, "-l", False])
