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
"""
cols_dict = {
    'balance-scale': ['target', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
    'banknote_authentication': ['variance-of-wavelet', 'skewness-of-wavelet', 'curtosis-of-wavelet', 'entropy',
                                'target'],
    'blood_transfusion': ['R', 'F', 'M', 'T', 'target'],
    'breast-cancer': ['target', 'age', 'menopause', 'tumor-size', 'inv-nodes',
                      'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'],
    'car': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'],
    'climate': ['Study', 'Run', 'vconst_corr', 'vconst_2', 'vconst_3', 'vconst_4', 'vconst_5', 'vconst_7', 'ah_corr',
                'ah_bolus', 'slm_corr', 'efficiency_factor', 'tidal_mix_max', 'vertical_decay_scale', 'convect_corr',
                'bckgrnd_vdc1', 'bckgrnd_vdc_ban', 'bckgrnd_vdc_eq', 'bckgrnd_vdc_psim', 'Prandtl', 'target'],
    'glass': ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'target'],
    'hayes-roth': ['file_name', 'hobby', 'age', 'educational-level', 'marital-status', 'target'],
    'house-votes-84': ['target', 'handicapped-infants', 'water-project-cost-sharing',
                       'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                       'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                       'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
                       'superfund-right-to-sue', 'crime', 'duty-free-exports',
                       'export-administration-act-south-africa'],
    'image_segmentation': ['target', 'region-centroid-col', 'region-centroid-row', 'region-pixel-count',
                           'short-line-density-5', 'short-line-density-2', 'vedge-mean', 'vegde-sd', 'hedge-mean',
                           'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean', 'rawgreen-mean', 'exred-mean',
                           'exblue-mean', 'exgreen-mean', 'value-mean', 'saturatoin-mean', 'hue-mean'],
    'ionosphere': list(range(1, 35)) + ['target'],
    'iris': ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target'],
    'kr-vs-kp': ['bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp', 'bxqsq', 'cntxt',
                 'dsopp','dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8', 'reskd', 'reskr', 'rimmx', 'rkxwp', 'rxmsq',
                 'simpl','skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk', 'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos',
                 'wtoeg','target'],
    'monk1': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
    'monk2': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
    'monk3': ['target', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
    'parkinsons': ['file_name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)','MDVP:RAP',
                   'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                   'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'target', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
    'soybean-small': list(range(1, 36)) + ['target'],
    'tic-tac-toe': list("a{}".format(j+1) for j in range(9)) + ['target'],
    'wine-red': ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
                 'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'target'],
    'wine-white': ['fixed-acidity', 'volatile-acidity', 'citric-acid', 'residual-sugar', 'chlorides',
                   'free-sulfur dioxide', 'total-sulfur-dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'target']
}
categorical_datasets = ['balance_scale','car','kr-vs-kp','house-votes-84','hayes-roth',
                            'monk1','monk2','monk3','soybean_small','tic-tac-toe']
numerical_datasets = ['iris','wine-red','wine-white','breast-cancer','banknote_authentication', 'blood_transfusion',
                          'climate','glass','image_segmentation','ionosphere','parkinsons']
for file_name in cols_dict:
    data = UTILS.get_data(file_name,'target')
"""