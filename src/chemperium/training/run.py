import pandas as pd
import keras
import tensorflow as tf
from chemperium.inp import InputArguments
from chemperium.data.load_test_data import TestInputArguments
from chemperium.data.load_data import *
from chemperium.model.mpnn import MPNN
from chemperium.model.ffn import NeuralNetwork
from chemperium.molecule.batch import MPNNDataset, prepare_batch
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy, mean_squared_error
from chemperium.training.evaluate import *
import gc
from typing import Union, List, Tuple
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler


def run_fp_model(
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        train_indices: npt.NDArray[np.int64],
        validation_indices: npt.NDArray[np.int64],
        inp: Union[InputArguments, TestInputArguments],
        model: Union[None, Model] = None
) -> Model:
    x_train = x[train_indices]
    x_val = x[validation_indices]

    y_train = y[train_indices]
    y_val = y[validation_indices]

    print(f"\nThere are {len(y_train)} training data points.\nThere are {len(y_val)} validation data points.\n")

    if len(y_train.shape) == 1:
        output_shape = 1
    else:
        output_shape = y_train.shape[1]

    if model is None:
        model = NeuralNetwork(input_size=x.shape[1], output_size=output_shape, inp=inp).model
        model = compile_model(model=model, inp=inp)

    es = EarlyStopping(patience=inp.patience, restore_best_weights=True, min_delta=0.00001, mode='min')
    model.summary()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=inp.max_epochs, callbacks=[es], verbose=2)

    return model


def run_model(x: Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor,
                       tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor,
                       tf.RaggedTensor, tf.RaggedTensor],
              y: npt.NDArray[np.float64],
              train_indices: npt.NDArray[np.int64],
              validation_indices: npt.NDArray[np.int64],
              inp: Union[InputArguments, TestInputArguments],
              model: Union[None, Model] = None) -> Model:
    """
    This function trains an individual message-passing neural network.
    :param x: Tuple of RaggedTensors
    :param y: NumPy array with output data
    :param train_indices: NumPy array with training indices
    :param validation_indices: NumPy array with validation indices
    :param inp: Input arguments
    :param model: (Optional) Pretrained Keras model
    :return: Trained Keras model
    """
    x_train = tuple(tf.gather(tup, train_indices) for tup in x)
    x_val = tuple(tf.gather(tup, validation_indices) for tup in x)

    y_train = y[train_indices]
    y_val = y[validation_indices]

    print(f"\nThere are {len(y_train)} training data points.\nThere are {len(y_val)} validation data points.\n")

    if len(y_train.shape) == 1:
        output_shape = 1
    else:
        output_shape = y_train.shape[1]

    x_valp, y_valp = prepare_batch(x_val, y_val)

    # print(f"Shape of atom features: {x_valp[0].shape}")
    # print(f"Shape of bond features: {x_valp[1].shape}")
    # print(f"Shape of H-bond features: {x_valp[2].shape}")

    if model is None:
        mpnn = MPNN(
            x_valp[0][0].shape[1],
            x_valp[1][0].shape[1],
            output_shape,
            hidden_message=inp.hidden_message,
            representation_size=inp.representation_size,
            depth=inp.depth,
            hidden_size=inp.hidden_size,
            layers=inp.num_layers,
            include_3d=inp.include_3d,
            mean_readout=inp.mean_readout,
            mfd=inp.mfd,
            seed=inp.seed,
            activation=inp.activation,
            dropout=inp.dropout,
            batch_normalization=inp.batch_normalization,
            l2_value=inp.l2
        )
        mpnn = compile_model(model=mpnn, inp=inp)
    else:
        mpnn = model

    if model is None:
        xt = MPNNDataset(x_train, y_train, batch_size=inp.batch_size, seed=inp.seed)
        xv = MPNNDataset(x_val, y_val, batch_size=inp.batch_size, seed=inp.seed)
        es = EarlyStopping(patience=inp.patience, restore_best_weights=True, min_delta=0.00001, mode='min')
        mpnn.summary()
    else:
        xt = MPNNDataset(x_train, y_train, batch_size=inp.transfer_batch, seed=inp.seed)
        xv = MPNNDataset(x_val, y_val, batch_size=inp.transfer_batch, seed=inp.seed)
        es = EarlyStopping(patience=inp.transfer_patience, restore_best_weights=True, min_delta=0.00001, mode='min')
        mpnn.summary()

    # xt, yt = prepare_batch(x_train, y_train)
    hist = mpnn.fit(xt, validation_data=xv, epochs=inp.max_epochs, callbacks=[es], verbose=2)

    return mpnn


def ensemble(x: Union[Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor,
                            tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor,
                            tf.RaggedTensor, tf.RaggedTensor], npt.NDArray[np.float32]],
             y: npt.NDArray[np.float64],
             model_indices: npt.NDArray[np.int64],
             inp: Union[InputArguments, TestInputArguments],
             pretrained_models: Union[None, List[Model]] = None) -> List[Model]:
    """
    This function trains an ensemble of message-passing neural networks.
    :param x: Tuple of RaggedTensors
    :param y: NumPy array with output data
    :param model_indices: NumPy array containing the indices that do not belong to the test set
    :param inp: Input arguments
    :param pretrained_models: (Optional) List with pre-trained Keras models.
    :return: List with trained models
    """
    kf = KFold(n_splits=inp.outer_folds, random_state=None, shuffle=False)
    kf.get_n_splits(model_indices)

    models = []
    idx = 0

    for train_ids, val_ids in kf.split(model_indices):
        train_indices = model_indices[train_ids]
        validation_indices = model_indices[val_ids]
        if pretrained_models is None:
            if inp.fingerprint is None:
                mpnn = run_model(x, y, train_indices, validation_indices, inp)
            else:
                mpnn = run_fp_model(x, y, train_indices, validation_indices, inp)
        else:
            if inp.fingerprint is None:
                mpnn = run_model(x, y, train_indices, validation_indices, inp, pretrained_models[idx])
            else:
                mpnn = run_fp_model(x, y, train_indices, validation_indices, inp, pretrained_models[idx])
        if inp.store_models:
            mpnn.save(f"{inp.save_dir}/model_{idx}.keras")
        idx += 1
        models.append(mpnn)

    return models


def run_training(dl: DataLoader,
                 inp: Union[InputArguments, TestInputArguments],
                 return_models: bool = False,
                 pretrained_models: Union[None, List[Model], Model] = None) -> Union[List[Model], Model, None]:
    """
    This function is used to train message-passing neural networks
    :param dl: DataLoader object containing the training dataset
    :param inp: Input arguments
    :param return_models: Whether to return trained models as output of this function
    :param pretrained_models: Pre-trained Keras models
    :return: (Optional) Trained Keras models
    """

    keras.utils.set_random_seed(inp.seed)
    x = dl.x
    y = dl.y
    train_indices, validation_indices, test_indices, model_indices = split_dataset(len(y), seed=inp.seed,
                                                                                   split_ratio=inp.ratio)
    if inp.ensemble:
        models = ensemble(x, y, model_indices, inp, pretrained_models=pretrained_models)
        if inp.ratio[2] != 0.0:
            df_pred = evaluate_ensemble(models, dl, test_indices, inp)
            df_pred.to_csv(inp.save_dir + "/test_preds.csv")
        df_val = evaluate_ensemble(models, dl, validation_indices, inp)
        df_val.to_csv(inp.save_dir + "/validation_preds.csv")
        if return_models:
            return models
        elif inp.test:
            test_external_dataset(models, dl.scaler, inp)
            return None
        return None
    else:
        if pretrained_models is None:
            if inp.fingerprint is None:
                mpnn = run_model(x, y, train_indices, validation_indices, inp)
            else:
                mpnn = run_fp_model(x, y, train_indices, validation_indices, inp)
        else:
            if inp.fingerprint is None:
                mpnn = run_model(x, y, train_indices, validation_indices, inp, pretrained_models)
            else:
                mpnn = run_fp_model(x, y, train_indices, validation_indices, inp, pretrained_models)

        if inp.store_models:
            mpnn.save(f"{inp.save_dir}/model.keras")
        if inp.ratio[2] != 0.0:
            df_pred = evaluate_training_model(mpnn, dl, test_indices, inp)
            df_pred.to_csv(inp.save_dir + "/test_preds.csv")
        df_val = evaluate_training_model(mpnn, dl, validation_indices, inp)
        df_val.to_csv(inp.save_dir + "/validation_preds.csv")
        if return_models:
            return mpnn

        elif inp.test:
            test_external_dataset(mpnn, dl.scaler, inp)
            return None
        return None


def run_pretraining(dl: DataLoader,
                    inp: Union[InputArguments, TestInputArguments]) -> Union[List[Model], Model]:
    models = run_training(dl, inp, True, None)
    return models


def run_transfer(dl_large: DataLoader,
                 inp: Union[InputArguments, TestInputArguments]) -> None:
    """
    This function performs transfer learning.
    :param dl_large: DataLoader containing the pre-training data
    :param inp: Input arguments
    :return: Function does not return anything
    """
    models = run_pretraining(dl_large, inp)

    del dl_large
    gc.collect()
    if inp.locked_transfer:
        if inp.ensemble:
            for model in models:
                for layer in model.layers[:9]:
                    layer.trainable = False
        elif type(models) is Model:
            for layer in models.layers[:9]:
                layer.trainable = False
        else:
            raise TypeError("Incorrect Keras model type.")

    print("Start loading the transfer file...")
    inp.input_file = inp.transfer_file
    inp.property = inp.transfer_property
    print(f"New input file: {inp.input_file}")
    print(f"Transfer file: {inp.transfer_file}")

    dl_small = DataLoader(input_pars=inp, transfer=True)
    if inp.test:
        fine_tuned_models = run_training(dl_small, inp, return_models=True, pretrained_models=models)
        test_external_dataset(fine_tuned_models, dl_small.scaler, inp)
    else:
        run_training(dl_small, inp, return_models=False, pretrained_models=models)


def test_external_dataset(models: Union[Model, List[Model]],
                          scaler: Union[None, MinMaxScaler],
                          inp: Union[InputArguments, TestInputArguments],
                          dl: Union[None, DataLoader] = None,
                          return_results: bool = False) -> Union[None, pd.DataFrame]:
    """
    This function makes predictions on an external dataset using trained Keras models.
    :param models: List of keras models or a single keras model
    :param scaler: MinMaxScaler to scale data
    :param inp: Input arguments
    :param dl: (Optional) Add pre-made DataLoader
    :param return_results: (Optional) Return results as pandas dataframe. Will by default write results to a file.
    :return: (Optional) Pandas DataFrame with predictions
    """

    if dl is None:
        print("Start loading the test file...")
        inp.input_file = inp.test_file
        dl_test = DataLoader(input_pars=inp, transfer=False, test=True)
        dl_test.scaler = scaler
    else:
        dl_test = dl
    if inp.ensemble:
        df_pred = external_ensemble_test(models, dl_test, inp)
        if not return_results:
            df_pred.to_csv(inp.save_dir + "/external_test_ensemble_preds.csv")
            return None
        else:
            return df_pred
    else:
        df_pred = external_model_test(models, dl_test, inp)
        if not return_results:
            df_pred.to_csv(inp.save_dir + "/external_test_preds.csv")
            return None
        else:
            return df_pred


def compile_model(model: Model, inp: Union[InputArguments, TestInputArguments]) -> Model:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        inp.init_lr,
        decay_steps=inp.decay_steps,
        decay_rate=inp.decay_rate,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=inp.clipvalue)

    if inp.activation == "linear":
        if inp.masked:
            model.compile(
                loss=masked_mean_squared_error,
                optimizer=optimizer,
                metrics=[masked_mean_absolute_error, 'mean_absolute_percentage_error']
            )
        else:
            model.compile(
                loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
            )
    elif inp.activation == "softmax":
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(from_logits=False),
            metrics=['categorical_accuracy']
        )
    elif inp.activation == "sigmoid":
        if inp.masked:
            model.compile(
                optimizer=optimizer,
                loss=masked_binary_crossentropy,
                metrics=['binary_accuracy', 'AUC']
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss=BinaryCrossentropy(from_logits=False),
                metrics=['binary_accuracy', 'AUC']
            )
    else:
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )

    return model


def masked_binary_crossentropy(y_true, y_pred):
    mask = ~tf.math.is_nan(y_true)  # Create a mask where True if y_true is not NaN
    masked_y_true = tf.boolean_mask(y_true, mask)  # Apply mask to ground truth
    masked_y_pred = tf.boolean_mask(y_pred, mask)  # Apply mask to predictions
    return tf.reduce_mean(BinaryCrossentropy(masked_y_true - masked_y_pred))  # Compute MSE


def masked_mean_squared_error(y_true, y_pred):
    mask = ~tf.math.is_nan(y_true)  # Create a mask where True if y_true is not NaN
    masked_y_true = tf.boolean_mask(y_true, mask)  # Apply mask to ground truth
    masked_y_pred = tf.boolean_mask(y_pred, mask)  # Apply mask to predictions
    return tf.reduce_mean(tf.square(masked_y_true - masked_y_pred))  # Compute MSE


def masked_mean_absolute_error(y_true, y_pred):
    mask = ~tf.math.is_nan(y_true)
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.abs(masked_y_true - masked_y_pred))
