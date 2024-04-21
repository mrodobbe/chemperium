from chemperium.inp import InputArguments
from chemperium.data.load_data import DataLoader, input_checker
from chemperium.training.run import run_training, run_transfer

if __name__ == "__main__":

    inputs = InputArguments()
    input_checker(inputs.save_dir)
    if inputs.transfer:
        print(f"Input file: {inputs.input_file}")
        print(f"Transfer file: {inputs.transfer_file}")
        dl_large = DataLoader(inputs)
        run_transfer(dl_large, inputs)
    else:
        dl = DataLoader(inputs)
        run_training(dl, inputs)
