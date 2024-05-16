from chemperium.inp import InputArguments
from chemperium.data.load_data import DataLoader
from chemperium.data.load_test_data import read_csv, load_models
from chemperium.training.run import test_external_dataset


if __name__ == "__main__":
    inp = InputArguments(training_type="test")
    df = read_csv(inp)
    models, scaler = load_models(inp)
    dl_test = DataLoader(input_pars=inp, transfer=False, test=True, df=df)
    dl_test.scaler = scaler
    test_external_dataset(models, scaler, inp, dl_test)
