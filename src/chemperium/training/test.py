from typing import Union, List, Dict, Any

import pandas as pd
from chemperium.inp import InputArguments
from chemperium.data.load_data import DataLoader
from chemperium.data.load_test_data import read_csv, load_models
from chemperium.training.run import test_external_dataset


def test(smiles: List[str],
         prop: Union[str, List[str]],
         save_dir: str,
         xyz: Union[List[str], None] = None,
         return_results: bool = False,
         input_args: Union[None, Dict[str, Any]] = None) -> Union[None, pd.DataFrame]:
    """
    Test custom message-passing neural networks
    :param smiles: SMILES to be tested
    :param prop: comma-separated string or list of strings with target property/ies.
    :param save_dir: folder to store data
    :param xyz: (optional) xyz coordinates of target compounds
    :param input_args: dictionary with input arguments. See inp.py for complete overview
    :return:
    """

    inputs = InputArguments(training_type="test")

    if input_args is not None:
        for key, value in input_args.items():
            setattr(inputs, key, value)

    if type(prop) is str:
        prop = prop.split(",")

    inputs.test = True

    setattr(inputs, "property", prop)
    setattr(inputs, "save_dir", save_dir)

    df = read_csv(inputs, smiles=smiles, xyz_list=xyz)
    models, scaler = load_models(inputs)
    dl_test = DataLoader(input_pars=inputs, transfer=False, test=True, df=df)
    dl_test.scaler = scaler
    res = test_external_dataset(models, scaler, inputs, dl_test, return_results=return_results)
    return res
