from typing import Union, List
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, r2_score
import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd
from chemperium.data.load_data import DataLoader
from chemperium.molecule.batch import prepare_batch
from chemperium.inp import InputArguments
from chemperium.data.load_test_data import TestInputArguments
import pickle
import tensorflow as tf
from keras.models import Model
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def evaluate_training_model(
        model: Model,
        dl: DataLoader,
        test_indices: npt.NDArray[np.int64],
        inp: Union[InputArguments, TestInputArguments]
) -> pd.DataFrame:

    x_test = tuple(tf.gather(tup, test_indices) for tup in dl.x)
    y_test = dl.y[test_indices]
    x_test, y_test = prepare_batch(x_test, y_test)

    smiles = dl.smiles[test_indices]

    if len(y_test.shape) > 1:
        test_predictions = np.asarray(model.predict([x_test])).astype(np.float32)
    else:
        test_predictions = model.predict([x_test]).reshape(-1)

    if inp.scaler is True:
        dl.scaler.inverse_transform(y_test)
        dl.scaler.inverse_transform(test_predictions)

    if inp.activation == "softmax" or inp.activation == "sigmoid":
        if len(y_test.shape) == 1:
            pred_dict = {
                "smiles": smiles,
                f"{inp.property[0]} pred": test_predictions,
                f"{inp.property[0]} true": y_test
            }
            df_pred = pd.DataFrame(pred_dict)

        else:
            pred_dict = {"smiles": smiles}
            df_pred = pd.DataFrame(pred_dict)
            for i in range(len(inp.property)):
                df_pred[f"{inp.property[i]} true"] = y_test[:, i]
                df_pred[f"{inp.property[i]} pred"] = test_predictions[:, i]

        acc = accuracy_score(y_test, (test_predictions > 0.5))
        f1 = f1_score(y_test, (test_predictions > 0.5), average='weighted')
        print(f"Accuracy: {acc:.3f}")
        print(f"F1 Score: {f1:.3f}")

        # Other metrics for classification
        report = classification_report(y_test, (test_predictions > 0.5))
        print(report)

    else:
        test_error = abs(test_predictions - y_test)
        mae = np.average(test_error)
        rmse = np.sqrt(np.average(test_error ** 2))
        r2 = r2_score(y_test, test_predictions)

        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")

        if len(y_test.shape) == 1:
            pred_dict = {f"{inp.property[0]} pred": test_predictions, f"{inp.property[0]} true": y_test,
                         "smiles": smiles, f"{inp.property[0]} error": test_error}
            df_pred = pd.DataFrame(pred_dict)
        else:
            if inp.property == "sigma":
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for j in range(y_test.shape[1]):
                    df_pred[f"sig_{j} true"] = y_test[:, j]
                for j in range(y_test.shape[1]):
                    df_pred[f"sig_{j} pred"] = test_predictions[:, j]
                for j in range(y_test.shape[1]):
                    df_pred[f"sig_{j} error"] = test_error[:, j]
            else:
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for i in range(len(inp.property)):
                    df_pred[f"{inp.property[i]} true"] = y_test[:, i]
                    df_pred[f"{inp.property[i]} pred"] = test_predictions[:, i]
                    df_pred[f"{inp.property[i]} error"] = test_error[:, i]

    return df_pred


def evaluate_ensemble(
        models: List[Model],
        dl: DataLoader,
        test_indices: npt.NDArray[np.int64],
        inp: Union[InputArguments, TestInputArguments]
) -> pd.DataFrame:

    x_test = tuple(tf.gather(tup, test_indices) for tup in dl.x)
    y_test = dl.y[test_indices]
    x_test, y_test = prepare_batch(x_test, y_test)
    smiles = dl.smiles[test_indices]

    ensemble = np.array([])
    for model in models:
        if len(y_test.shape) > 1:
            test_predictions = np.asarray(model.predict([x_test])).astype(np.float32)
        else:
            test_predictions = model.predict([x_test]).reshape(-1)
        test_predictions = tf.constant(test_predictions)[tf.newaxis, :]
        if len(ensemble) == 0:
            ensemble = test_predictions
        else:
            ensemble = np.vstack((ensemble, test_predictions))

    ensemble_predictions = np.array(ensemble)

    prediction = np.mean(ensemble_predictions, axis=0)
    if inp.scaler:
        dl.scaler.inverse_transform(y_test)
        dl.scaler.inverse_transform(prediction)
    sd = np.std(ensemble_predictions, axis=0)
    if inp.scaler is True:
        sd = sd * dl.scaler.data_range_

    if inp.activation == "softmax" or inp.activation == "sigmoid":

        if len(y_test.shape) == 1:
            pred_dict = {
                "smiles": smiles,
                f"{inp.property[0]} pred": prediction,
                f"{inp.property[0]} true": y_test,
                f"{inp.property[0]} sd": sd
            }

            df_pred = pd.DataFrame(pred_dict)
        else:
            pred_dict = {"smiles": smiles}
            df_pred = pd.DataFrame(pred_dict)
            for i in range(len(inp.property)):
                df_pred[f"{inp.property[i]} true"] = y_test[:, i]
                df_pred[f"{inp.property[i]} pred"] = prediction[:, i]
                df_pred[f"{inp.property[i]} sd"] = sd[:, i]

    else:
        abs_error = np.abs(prediction - y_test)
        mae = np.average(abs_error)
        rmse = np.sqrt(np.average(abs_error ** 2))
        r2 = r2_score(y_test, prediction)

        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")

        if len(y_test.shape) == 1:
            pred_dict = {f"{inp.property[0]} pred": prediction, f"{inp.property[0]} true": y_test,
                         "smiles": smiles, f"{inp.property[0]} error": abs_error, f"{inp.property[0]} sd": sd}
            df_pred = pd.DataFrame(pred_dict)
        else:
            if "sigma" in inp.property:
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for j in range(y_test.shape[1]):
                    df_pred[f"sig_{j} true"] = y_test[:, j]
                    df_pred[f"sig_{j} pred"] = prediction[:, j]
                    df_pred[f"sig_{j} error"] = abs_error[:, j]
                    df_pred[f"sig_{j} sd"] = sd[:, j]
            else:
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for i in range(len(inp.property)):
                    df_pred[f"{inp.property[i]} true"] = y_test[:, i]
                    df_pred[f"{inp.property[i]} pred"] = prediction[:, i]
                    df_pred[f"{inp.property[i]} error"] = abs_error[:, i]
                    df_pred[f"{inp.property[i]} sd"] = sd[:, i]

    return df_pred


def external_ensemble_test(models: List[Model],
                           dl: DataLoader,
                           inp: Union[InputArguments, TestInputArguments]) -> pd.DataFrame:
    x_test = dl.x
    smiles = dl.smiles
    x_test, y_test = prepare_batch(x_test, [])

    # Determine output shape for predictions
    if inp.property == "sigma":
        output_shape = 55
    else:
        output_shape = len(inp.property)

    ensemble = np.array([])
    for model in models:
        if output_shape > 1:
            test_predictions = np.asarray(model.predict([x_test], verbose=0)).astype(np.float32)
        else:
            test_predictions = np.asarray(model([x_test])).reshape(-1)
        test_predictions = tf.constant(test_predictions)[tf.newaxis, :]
        if len(ensemble) == 0:
            ensemble = test_predictions
        else:
            ensemble = np.vstack((ensemble, test_predictions))

    prediction = np.mean(ensemble, axis=0)
    if output_shape == 1:
        prediction = prediction.reshape(-1, 1)
    if inp.scaler:
        dl.scaler.inverse_transform(prediction)

    sd = np.std(ensemble, axis=0)
    if inp.scaler is True:
        sd = sd * dl.scaler.data_range_

    if inp.activation == "softmax" or inp.activation == "sigmoid":
        if output_shape == 1:
            prediction = prediction.reshape(-1)
            pred_dict = {f"smiles": smiles,
                         f"{inp.property[0]}_prediction": prediction, f"{inp.property[0]}_uncertainty": sd}
            df_pred = pd.DataFrame(pred_dict)
        else:
            pred_dict = {"smiles": smiles}
            df_pred = pd.DataFrame(pred_dict)
            for i in range(output_shape):
                df_pred[f"{inp.property[i]}_prediction"] = prediction[:, i]
                df_pred[f"{inp.property[i]}_uncertainty"] = sd[:, i]

    else:
        if output_shape == 1:
            prediction = prediction.reshape(-1)
            pred_dict = {f"smiles": smiles,
                         f"{inp.property[0]}_prediction": prediction, f"{inp.property[0]}_uncertainty": sd}
            df_pred = pd.DataFrame(pred_dict)
        else:
            if "sigma" in inp.property:
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for j in range(output_shape):
                    df_pred[f"sig_{j}_prediction"] = prediction[:, j]
                    df_pred[f"sig_{j}_uncertainty"] = sd[:, j]
            else:
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for i in range(output_shape):
                    df_pred[f"{inp.property[i]}_prediction"] = prediction[:, i]
                    df_pred[f"{inp.property[i]}_uncertainty"] = sd[:, i]

    return df_pred


def external_model_test(model: Model,
                        dl: DataLoader,
                        inp: Union[InputArguments, TestInputArguments]) -> pd.DataFrame:
    x_test = dl.x
    smiles = dl.smiles
    x_test, y_test = prepare_batch(x_test, [])

    if inp.property == "sigma":
        output_shape = 55
    else:
        output_shape = len(inp.property)

    test_predictions = np.asarray(model.predict([x_test])).astype(np.float32)
    if inp.scaler:
        dl.scaler.inverse_transform(test_predictions)

    if inp.activation == "softmax" or inp.activation == "sigmoid":
        if output_shape == 1:
            test_predictions = test_predictions.reshape(-1)
            pred_dict = {
                "smiles": smiles,
                f"{inp.property[0]}_prediction": test_predictions
            }
            df_pred = pd.DataFrame(pred_dict)

        else:
            pred_dict = {"smiles": smiles}
            df_pred = pd.DataFrame(pred_dict)
            for i in range(len(inp.property)):
                df_pred[f"{inp.property[i]}_prediction"] = test_predictions[:, i]

    else:
        if output_shape == 1:
            test_predictions = test_predictions.reshape(-1)
            pred_dict = {"smiles": smiles, f"{inp.property[0]}_prediction": test_predictions}
            df_pred = pd.DataFrame(pred_dict)
        else:
            if inp.property == "sigma":
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for j in range(output_shape):
                    df_pred[f"sig_{j}_prediction"] = test_predictions[:, j]
            else:
                pred_dict = {"smiles": smiles}
                df_pred = pd.DataFrame(pred_dict)
                for i in range(output_shape):
                    df_pred[f"{inp.property[i]}_prediction"] = test_predictions[:, i]

    return df_pred
