from chemperium.data.load_test_data import load_models, TestInputArguments, read_csv
from chemperium.data.load_data import DataLoader
from chemperium.molecule.batch import prepare_batch
from typing import Union, List, Tuple
import numpy as np
import numpy.typing as npt
from keras.models import Model


class Representation:
    """
    This class creates a learned fingerprint for organic molecules.
    Currently, the only available fingerprint is pretrained on the COSMO-RS properties in ReagLib20 and DrugLib36.
    """
    def __init__(self,
                 data_location: Union[str, None] = None):

        self.inputs = TestInputArguments("2d")

        self.inputs.property = ['Tb', 'cP', 'cT', 'cV', 'cTb',
                                'Area', 'sig1', 'sig2', 'sig3',
                                'sig4', 'sig5', 'sig6', 'E_vdw',
                                'av_dE', 'E_diel', 'E_ring', 'Volume',
                                'area_n', 'area_p', 'mu_gas',
                                'Hb_acc1', 'Hb_acc2', 'Hb_acc3', 'Hb_acc4', 'Hb_acc5',
                                'Hb_don1', 'Hb_don2', 'Hb_don3', 'Hb_don4', 'Hb_don5',
                                'charge_n', 'charge_p', 'log(KOC)', 'logP(BB)',
                                'Abraham_A', 'Abraham_B', 'Abraham_E', 'Abraham_S', 'Abraham_Vx',
                                'logS(Water)', 'Hsolv(Water)', 'logP(CS2-Water)', 'logP(Gas-Water)',
                                'logK(Cell-Water)', 'logP(CCl4-Water)', 'logP(Et2O-Water)',
                                'logP(PGDP-Water)', 'logP(Skin-Water)', 'Percent-IntAbsorb',
                                'logP(CHCl3-Water)', 'logP(CH2Cl2-Water)', 'logP(Hexane-Water)',
                                'logP(Acetone-Water)', 'logP(Benzene-Water)', 'logP(Heptane-Water)',
                                'logP(Toluene-Water)', 'logP(OliveOil-Water)', 'logP(Isooctane-Water)',
                                'logP(Hexadecane-Water)', 'logP(Cyclohexane-Water)', 'logP(Octanol-dry-Water)',
                                'logP(Octanol-wet-Water)', 'logP(OleylAlcohol-Water)']

        self.inputs.ensemble = False
        if data_location is None:
            self.inputs.save_dir = self.inputs.dir + f"/cosmo/2d"
        else:
            self.inputs.save_dir = data_location + f"/caesar-data/cosmo/2d"
        self.models, self.scaler = load_models(self.inputs)

    def predict(self,
                smiles: Union[str, List[str]]) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.float64]]:
        """
        Create a learned fingerprint with the predict function
        :param smiles: A SMILES or a list of SMILES of the target molecule(s)
        :return: Returns a tuple with the parsed SMILES and their respective fingerprints
        """
        if type(smiles) is str:
            df_out = read_csv(self.inputs, smiles=[smiles])
        elif type(smiles) is list:
            df_out = read_csv(self.inputs, smiles=smiles)
        else:
            raise IndexError(f"Type {type(smiles)} is not supported for parameter smiles.")

        dl_test = DataLoader(input_pars=self.inputs, transfer=False, test=True, df=df_out)
        dl_test.scaler = self.scaler

        layer_name = 'readout'
        for layer in self.models[0].layers:
            if "readout" in layer.name:
                layer_name = layer.name

        intermediate_layer_model = Model(inputs=self.models[0].input,
                                         outputs=self.models[0].get_layer(layer_name).output)

        x_test, y_test = prepare_batch(dl_test.x, [])
        test_predictions = np.asarray(intermediate_layer_model.predict([x_test])).astype(np.float64)

        return dl_test.smiles, test_predictions
