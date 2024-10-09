from chemperium.inp import InputArguments
from chemperium.data.load_data import DataLoader, input_checker
from chemperium.training.run import run_training
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if __name__ == "__main__":
    inputs = InputArguments()
    input_checker(inputs.save_dir)
    results_dict = {"Layers": [], "Neurons": [], "Init_LR": [], "Activation": [], "Dropout": []}
    for j in range(len(inputs.property)):
        prop = inputs.property[j]
        results_dict[f"{prop}_MAE"] = []
        results_dict[f"{prop}_RMSE"] = []
        results_dict[f"{prop}_R2"] = []
    dfs = []
    num_layers = [2, 3, 4, 5, 6]
    hidden_size = [64, 128, 256, 512]
    init_lr = [1e-3, 1e-4]
    activations = ["LeakyReLU", "ReLU", "swish"]
    dropout = [0, 0.25, 0.5]
    for n in num_layers:
        for h in hidden_size:
            for i in init_lr:
                for a in activations:
                    for d in dropout:
                        print(n, "layers -", h, "neurons -", i, "init_lr -", a, "activation -", d, "dropout")
                        inputs.patience = 50
                        inputs.ensemble = False
                        inputs.num_layers = n
                        inputs.hidden_size = h
                        inputs.hidden_activation = a
                        inputs.init_lr = i
                        inputs.dropout = d
                        inputs.gmm_file = inputs.save_dir + "/gmm_dictionary.pickle"
                        inputs.ratio = (0.8, 0.1, 0.1)
                        dl = DataLoader(inputs)
                        run_training(dl, inputs)
                        df_pred = pd.read_csv(inputs.save_dir + "/test_preds.csv", index_col=0)
                        results_dict["Layers"].append(n)
                        results_dict["Neurons"].append(h)
                        results_dict["Init_LR"].append(i)
                        results_dict["Activation"].append(a)
                        results_dict["Dropout"].append(d)
                        for j in range(len(inputs.property)):
                            y_test = df_pred[f"{inputs.property[j]}_true"].to_numpy()
                            test_predictions = df_pred[f"{inputs.property[j]}_prediction"].to_numpy()
                            mae = mean_absolute_error(y_test, test_predictions)
                            rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                            r2 = r2_score(y_test, test_predictions)
                            results_dict[f"{inputs.property[j]}_MAE"].append(mae)
                            results_dict[f"{inputs.property[j]}_RMSE"].append(rmse)
                            results_dict[f"{inputs.property[j]}_R2"].append(r2)

                        df = pd.DataFrame(results_dict)
                        df.to_csv(inputs.save_dir + "/hyperparameter_optimization_results.csv")
