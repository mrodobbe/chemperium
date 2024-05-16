from typing import Union, List, Dict, Any
from chemperium.data.load_test_data import TestInputArguments
from chemperium.data.load_data import DataLoader, input_checker
from chemperium.training.run import run_training, run_transfer


def train(input_file: str,
          prop: Union[str, List[str]],
          save_dir: str,
          input_args: Union[None, Dict[str, Any]] = None) -> None:
    """
    Train custom message-passing neural networks
    :param input_file: string with the location of the csv file with input data
    :param prop: comma-separated string or list of strings with target property/ies.
    :param save_dir: folder to store data
    :param input_args: dictionary with input arguments. See inp.py for complete overview
    :return:
    """

    inputs = TestInputArguments()
    inputs.test = False
    inputs.ensemble = False
    setattr(inputs, "input_file", input_file)

    if input_args is not None:
        for key, value in input_args.items():
            setattr(inputs, key, value)

    if type(prop) is str:
        prop = prop.split(",")

    setattr(inputs, "property", prop)
    setattr(inputs, "save_dir", save_dir)

    if inputs.transfer_file is not "":
        inputs.transfer = True

    if inputs.test_file is not "":
        inputs.test = True

    if inputs.outer_folds > 1:
        inputs.ensemble = True

    input_checker(inputs.save_dir)
    dl = DataLoader(inputs)
    if inputs.transfer:
        run_transfer(dl, inputs)
    else:
        run_training(dl, inputs)

    print(f"FINISHED! The results are written in {inputs.save_dir}.")
