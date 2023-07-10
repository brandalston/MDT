using LinearAlgebra
using JSON
using StatsBase
using Random

function tree_eval(tree, X, D, m)
    n, p = size(X)
    y0 = zeros(n, m)
    if length(tree) != 4 || typeof(tree[3]) == typeof(1.0)
        return y0 .+ reshape(tree, (1,m))
    end
    f = tree[1]
    b = tree[2]
    idx1, idx2 = treesplit(x -> x<b, X[:,f])
    y0[idx1,:] = tree_eval(tree[3], X[idx1,:], D-1, m)
    y0[idx2,:] = tree_eval(tree[4], X[idx2,:], D-1, m)
    return y0    
end

function generate_realdata(name, seed=0)
    cols_dict = Dict(
        "auto-mpg" => ["target", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model-year",
        "origin", "car_name"],
        "balance_scale" => ["target", "left-weight", "left-distance", "right-weight", "right-distance"],
        "banknote" => ["variance-of-wavelet", "skewness-of-wavelet", "curtosis-of-wavelet", "entropy", "target"],
        "blood" => ["R", "F", "M", "T", "target"],
        "breast_cancer" => ["target", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"],
        "car" => ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"],
        "climate" => ["Study", "Run", "vconst_corr", "vconst_2", "vconst_3", "vconst_4", "vconst_5", "vconst_7",
                       "ah_corr", "ah_bolus", "slm_corr", "efficiency_factor", "tidal_mix_max", "vertical_decay_scale",
                       "convect_corr", "bckgrnd_vdc1", "bckgrnd_vdc_ban", "bckgrnd_vdc_eq", "bckgrnd_vdc_psim",
                       "Prandtl", "target"],
        "flare1" => ["class", "largest-spot-size", "spot-distribution", "activity", "evolution",
                      "previous-24hr-activity", "historically-complex", "become-h-c", "area", "area-largest-spot",
                      "c-target", "m-target", "x-target"],
        "flare2" => ["class", "largest-spot-size", "spot-distribution", "activity", "evolution",
                      "previous-24hr-activity", "historically-complex", "become-h-c", "area", "area-largest-spot",
                      "c-target", "m-target", "x-target"],
        "glass" => ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "target"],
        "hayes_roth" => ["name", "hobby", "age", "educational-level", "marital-status", "target"],
        "house_votes_84" => ["target", "handicapped-infants", "water-project-cost-sharing",
                              "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
                              "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
                              "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
                              "superfund-right-to-sue", "crime", "duty-free-exports",
                              "export-administration-act-south-africa"],
        "image" => ["target", "region-centroid-col", "region-centroid-row", "region-pixel-count",
                                  "short-line-density-5", "short-line-density-2", "vedge-mean", "vegde-sd",
                                  "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean",
                                  "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean",
                                  "saturatoin-mean", "hue-mean"],
        "ionosphere" => [1:35;"target"],
        "iris" => ["sepal-length", "sepal-width", "petal-length", "petal-width", "target"],
        "kr_vs_kp" => ["bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp", "bxqsq",
                        "cntxt", "dsopp", "dwipd", "hdchk", "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr",
                        "rimmx", "rkxwp", "rxmsq", "simpl", "skach", "skewr", "skrxp", "spcop", "stlmt", "thrsk",
                        "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg", "target"],
        "monk1" => ["target", "a1", "a2", "a3", "a4", "a5", "a6"],
        "monk2" => ["target", "a1", "a2", "a3", "a4", "a5", "a6"],
        "monk3" => ["target", "a1", "a2", "a3", "a4", "a5", "a6"],
        "parkinsons" => ["name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                          "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
                          "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "target", "RPDE",
                          "DFA", "spread1", "spread2", "D2", "PPE"],
        "soybean_small" => [1:36;"target"],
        "tic_tac_toe" => ["top-left-square", "top-middle-square", "top-right-square", "middle-left-square",
                           "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square",
                           "bottom-right-square", "target"],
        "wine_red" => ["fixed-acidity", "volatile-acidity", "citric-acid", "residual-sugar", "chlorides",
                        "free-sulfur dioxide", "total-sulfur-dioxide", "density", "pH", "sulphates", "alcohol",
                        "target"],
        "wine_white" => ["fixed-acidity", "volatile-acidity", "citric-acid", "residual-sugar", "chlorides",
                          "free-sulfur dioxide", "total-sulfur-dioxide", "density", "pH", "sulphates", "alcohol",
                          "target"]
    )

    Random.seed!(seed)

    data = DataFrame(CSV.File(pwd()*"/Datasets/"*name*".data"; header=cols_dict[name]))
    y_counter = 0
    y_encode = Dict()
    for i in unique!(data[:,:target])
        y_encode[i] = y_counter
    end

    features = filter(e->!(e in ["target"]), names(data))
    fullX, fullY = Matrix(data[:,features]), data[:,:target]

    n, p = size(fullX)
    goodfeature = Vector{Int64}()
    for i = 1:p
        if length(unique(fullX[:,i])) >= 2
            append!(goodfeature, i)
        end
    end
    gfullX = fullX[:,goodfeature]

    scaler = fit(UnitRangeTransform, gfullX, dims=1)
    scaledX = StatsBase.transform(scaler, gfullX)

    full_index = 1:size(data)[1]
    train_index = shuffle(full_index)[1:floor(Int, 0.5*size(data)[1])]
    poop_index = filter(e->!(e in train_index),full_index)
    cal_index = shuffle(poop_index)[1:floor(Int, 0.5*size(poop_index)[1])]
    test_index = filter(e->!(e in [train_index;cal_index]),full_index)

    X_train = zeros(size([train_index;cal_index])[1], size(features)[1])
    X_test = zeros(size(test_index)[1], size(features)[1])
    Y_train = zeros(size([train_index;cal_index])[1], size(unique!(data[:,:target]))[1])
    Y_test = zeros(size(test_index)[1], size(unique!(data[:,:target]))[1])


    counter = 1
    for i in [train_index;cal_index]
        X_train[counter,:] = scaledX[i,:]
        Y_train[counter, Int(y_encode[fullY[i]])+1] = 1
        counter += 1
    end

    counter = 1
    for i in test_index
        X_test[counter,:] = scaledX[i,:]
        Y_test[counter, Int(y_encode[fullY[i]])+1] = 1
        counter += 1
    end

    return X_train, X_test, Y_train, Y_test
end
