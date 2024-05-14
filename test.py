from src.chemperium import InputArguments
from src.chemperium import DataLoader
from src.chemperium.data.load_test_data import read_csv, load_models
from src.chemperium.training.run import test_external_dataset


if __name__ == "__main__":
    inp = InputArguments(training_type="test")
    df = read_csv(inp)
    models, scaler = load_models(inp)
    dl_test = DataLoader(input_pars=inp, transfer=False, test=True, df=df)
    dl_test.scaler = scaler
    test_external_dataset(models, scaler, inp, dl_test)
