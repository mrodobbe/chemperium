from typing import Union
from chemperium.inp import InputArguments
from chemperium.data.load_data import DataLoader, input_checker
from chemperium.training.run import run_training, run_transfer


def train(input_file: str, prop: Union[str, list], save_dir: str, input_args: dict = None):
    inputs = InputArguments()
    setattr(inputs, "input_file", input_file)
    for key, value in input_args.items():
        setattr(inputs, key, value)

    if type(prop) is str:
        prop = prop.split(",")

    setattr(inputs, "prop", prop)
    setattr(inputs, "save_dir", save_dir)

    if inputs.transfer_file is not None:
        inputs.transfer = True

    if inputs.test_file is not None:
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
