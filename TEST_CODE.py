import model_runs

'''
# MODEL RUN EXAMPLE
data_names = ['soybean-small','monk1','monk3','monk2','house-votes-84',
              'hayes-roth','breast-cancer','balance-scale','spect',
              'tic-tac-toe','kr-vs-kp','car_evaluation','fico_binary']
heights = [4]
models = ['CART']
time_limit = 3600
extras = ['max_features-25']
rand_states = [138, 15, 89, 42, 0]
tuning = None
file = 'results.csv'
plot_fig = False
consol_log = False
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models, "-t", time_limit, "-e", extras, "-r", rand_states, "-c", tuning,
     "-f", file, "-p", plot_fig, "-l", consol_log])
'''

time_limit = 600
rand_states = [138]
file = 'testing.csv'
heights = [3]
data_names = ['house-votes-84']
models = ['CUT1-ALL']
warm_start = {'use': False, 'values': None}
model_runs.main(
    ["-d", data_names, "-h", heights, "-m", models,
     "-t",time_limit, "-r", rand_states, "-w", warm_start, "-f", file, "-l", False])
