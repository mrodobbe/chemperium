from chemperium.data.load_data import DataLoader
from chemperium.data.load_test_data import read_csv, load_models, TestInputArguments
from chemperium.training.run import test_external_dataset
from chemperium.postprocessing.uncertainty_quantification import add_reliability_score
from chemperium.postprocessing.thermodynamics import *
from typing import Union
import pandas as pd
import numpy as np


class Thermo:
    """:class:`Thermo` contains trained models to predict the enthalpy of formation at 298 K."""
    def __init__(self, method: str, dimension: str, data_location: Union[str, None] = None):
        """
        :param method: A string with the quantum chemical method (currently supported: `g3mp2b3` and `cbs-qb3`)
        :param dimension: A string with the dimension of chemical information (`2d` or `3d`)
        """

        self.property = "Dh298"
        self.dimension = dimension
        self.inputs = TestInputArguments(dimension=self.dimension)
        self.inputs.property = ['H298_residual', 'S298',
                                'cp_25', 'cp_30', 'cp_40', 'cp_50',
                                'cp_60', 'cp_70', 'cp_80', 'cp_90',
                                'cp_100', 'cp_110', 'cp_120', 'cp_130',
                                'cp_140', 'cp_150', 'cp_175', 'cp_200',
                                'cp_225', 'cp_250', 'cp_275', 'cp_300',
                                'cp_325', 'cp_350', 'cp_375', 'cp_400',
                                'cp_425', 'cp_450', 'cp_475', 'cp_500',
                                'cp_525', 'cp_550', 'cp_575', 'cp_600',
                                'cp_650', 'cp_700', 'cp_750', 'cp_800',
                                'cp_850', 'cp_900', 'cp_950',
                                'cp_1000', 'cp_1050', 'cp_1100',
                                'cp_1150', 'cp_1200', 'cp_1250']
        if self.dimension == "2d":
            self.inputs.property[0] = 'H298'

        self.method = method
        if data_location is None:
            self.inputs.save_dir = self.inputs.dir + f"/caesar-data/thermo/{self.method}/{self.dimension}"
        else:
            self.inputs.save_dir = data_location + f"/thermo/{self.method}/{self.dimension}"
        self.models, self.scaler = load_models(self.inputs)

    def predict_enthalpy(self, smiles: Union[str, list], xyz: Union[str, list, None] = None,
                         llot: Union[float, list, None] = None, t: float = 298.15,
                         quality_check: bool = False) -> pd.DataFrame:
        """
        This function predicts a thermochemical property for the corresponding SMILES.
        :param xyz: Either a string of the xyz coordinates or a list of xyz coordinate strings.
        :param smiles: Either a SMILES string or a list of SMILES strings.
        :param llot: Low level-of-theory enthalpy of formation estimate
        :param t: Temperature at which the enthalpy of formation is estimated (in Kelvin)
        :param quality_check: Add a reliability score (default is False)
        :return: Predictions are returned as a Pandas DataFrame
        """

        df_pred, dl_test, llot = control_thermo_input(smiles, self.dimension, xyz, llot, t,
                                                      self.models, self.scaler, self.inputs, "h")

        if t == 298.15:
            if self.dimension == "2d":
                df_output = df_pred[["smiles", "H298_prediction", "H298_uncertainty"]].copy()
                if quality_check:
                    df_output[f"reliability"] = add_reliability_score(df_output["smiles"].tolist(),
                                                                      self.inputs.save_dir,
                                                                      df_output[f"H298_uncertainty"].tolist(),
                                                                      dl_test.df["RDMol"].tolist())
            elif llot is None:
                print(f"WARNING! Returning the residual between {self.method} and B3LYP in kcal/mol!")
                df_output = df_pred[["smiles", "H298_residual_prediction", "H298_residual_uncertainty"]].copy()
                if quality_check:
                    df_output[f"reliability"] = add_reliability_score(df_output["smiles"].tolist(),
                                                                      self.inputs.save_dir,
                                                                      df_output[f"H298_residual_uncertainty"].tolist(),
                                                                      dl_test.df["RDMol"].tolist())
            else:
                df_pred["H298_prediction"] = df_pred["H298_residual_prediction"].to_numpy() + llot
                df_pred["H298_uncertainty"] = df_pred["H298_residual_uncertainty"].to_numpy()
                df_output = df_pred[["smiles", "H298_prediction", "H298_uncertainty"]].copy()
                if quality_check:
                    df_output[f"reliability"] = add_reliability_score(df_output["smiles"].tolist(),
                                                                      self.inputs.save_dir,
                                                                      df_output[f"H298_uncertainty"].tolist(),
                                                                      dl_test.df["RDMol"].tolist())

        else:
            temperatures, cp_keys = get_heat_capacity_values(self.inputs)
            t_c = int(t)
            s298 = df_pred["S298_prediction"].to_numpy()
            cp = df_pred[cp_keys].to_numpy()

            if self.dimension == "2d":
                h298 = df_pred["H298_prediction"].to_numpy()
                unc = df_pred["H298_uncertainty"].to_numpy()

            else:
                h298 = df_pred["H298_residual_prediction"].to_numpy() + llot
                unc = df_pred["H298_residual_uncertainty"].to_numpy()

            nasa_coefficients = get_nasa_coefficients(temperatures, h298=h298, s298=s298, cp_values=cp)
            a1, a2, a3, a4, a5, a6, a7 = nasa_coefficients.T
            h = enthalpy_fit(t, a1, a2, a3, a4, a5, a6) * 8.314 / 4.184
            df_output = df_pred[["smiles"]].copy()
            df_output[f"H{t_c}_prediction"] = h
            df_output[f"H{t_c}_uncertainty"] = unc
            if quality_check:
                df_output[f"reliability"] = add_reliability_score(df_output["smiles"].tolist(),
                                                                  self.inputs.save_dir,
                                                                  unc,
                                                                  dl_test.df["RDMol"].tolist())

        return df_output

    def predict_entropy(self, smiles: Union[str, list], xyz: Union[str, list, None] = None,
                        t: float = 298.15, quality_check: bool = False):
        df_pred, dl_test, llot = control_thermo_input(smiles, self.dimension, xyz, None, t,
                                                      self.models, self.scaler, self.inputs, "s")

        if t == 298.15:
            df_output = df_pred[["smiles", "S298_prediction", "S298_uncertainty"]].copy()
            if quality_check:
                df_output[f"reliability"] = add_reliability_score(df_output["smiles"].tolist(),
                                                                  self.inputs.save_dir,
                                                                  df_output[f"S298_uncertainty"].tolist(),
                                                                  dl_test.df["RDMol"].tolist())

        else:
            temperatures, cp_keys = get_heat_capacity_values(self.inputs)
            s298 = df_pred["S298_prediction"].to_numpy()

            a1, a2, a3, a4, a5 = get_cp_coefficients(temperatures, df_pred[cp_keys].to_numpy() * (4.184 / 8.314))
            a7 = s298 * (4.184 / 8.314) - entropy_fit(298.15, a1, a2, a3, a4, a5, 0)
            s = entropy_fit(t, a1, a2, a3, a4, a5, a7) * 8.314 / 4.184

            df_output = df_pred[["smiles"]].copy()
            t_c = int(t)
            df_output[f"S{t_c}_prediction"] = s
            df_output[f"S{t_c}_uncertainty"] = df_pred["S298_uncertainty"].to_numpy()
            if quality_check:
                df_output[f"reliability"] = add_reliability_score(df_output["smiles"].tolist(),
                                                                  self.inputs.save_dir,
                                                                  df_output[f"S{t_c}_uncertainty"].tolist(),
                                                                  dl_test.df["RDMol"].tolist())

        return df_output

    def predict_gibbs(self, smiles: Union[str, list], xyz: Union[str, list, None] = None,
                      llot: Union[float, list, None] = None, t: float = 298.15,
                      quality_check: bool = False) -> pd.DataFrame:
        df_pred, dl_test, llot = control_thermo_input(smiles, self.dimension, xyz, llot, t,
                                                      self.models, self.scaler, self.inputs, "g")
        t_c = int(t)
        df_enthalpy = self.predict_enthalpy(smiles, xyz, llot, t, quality_check)
        df_entropy = self.predict_entropy(smiles, xyz, t, quality_check)

        df_gibbs = df_enthalpy.copy()
        df_gibbs[f"S{t_c}_prediction"] = df_entropy[f"S{t_c}_prediction"].to_numpy()
        df_gibbs[f"S{t_c}_uncertainty"] = df_entropy[f"S{t_c}_uncertainty"].to_numpy()
        h = df_gibbs[f"H{t_c}_prediction"].to_numpy()
        s = df_gibbs[f"S{t_c}_prediction"].to_numpy()
        df_gibbs[f"G{t_c}_prediction"] = h - t * 0.001 * s

        if quality_check:
            df_gibbs[f"H{t_c}_reliability"] = df_enthalpy["reliability"].tolist()
            df_gibbs[f"S{t_c}_reliability"] = df_entropy["reliability"].tolist()

            df_output = df_gibbs[["smiles", f"G{t_c}_prediction",
                                  f"H{t_c}_prediction", f"S{t_c}_prediction",
                                  f"H{t_c}_uncertainty", f"S{t_c}_uncertainty",
                                  f"H{t_c}_reliability", f"S{t_c}_reliability"]]
        else:
            df_output = df_gibbs[["smiles", f"G{t_c}_prediction",
                                  f"H{t_c}_prediction", f"S{t_c}_prediction",
                                  f"H{t_c}_uncertainty", f"S{t_c}_uncertainty"]]

        return df_output

    def get_nasa_polynomials(self, names: Union[str, list], smiles: Union[str, list],
                             xyz: Union[str, list, None] = None,
                             llot: Union[float, list, None] = None, chemkin: bool = False):
        """
        This function predicts the thermochemistry for the corresponding SMILES and returns the NASA polynomials.
        :param names: Either a string or a list of strings containing the species names.
        :param xyz: Either a string of the xyz coordinates or a list of xyz coordinate strings.
        :param smiles: Either a SMILES string or a list of SMILES strings.
        :param llot: Low level-of-theory enthalpy of formation estimate
        :param chemkin: Whether to return a Chemkin thermo file
        """

        df_pred, dl_test, llot = control_thermo_input(smiles, self.dimension, xyz, llot, 300,
                                                      self.models, self.scaler, self.inputs, "h")
        mols = dl_test.df["RDMol"].tolist()
        temperatures, cp_keys = get_heat_capacity_values(self.inputs)

        if self.dimension == "2d":
            nasa_coefficients = get_nasa_coefficients(temperatures, h298=df_pred["H298_prediction"].to_numpy(),
                                                      s298=df_pred["S298_prediction"].to_numpy(),
                                                      cp_values=df_pred[cp_keys].to_numpy())

        else:
            h298 = df_pred["H298_residual_prediction"].to_numpy() + llot
            nasa_coefficients = get_nasa_coefficients(temperatures, h298=h298,
                                                      s298=df_pred["S298_prediction"].to_numpy(),
                                                      cp_values=df_pred[cp_keys].to_numpy())

        if chemkin:
            if type(names) is str:
                chemkin_data = get_chemkin_file(name=names, smiles=smiles, method=self.method,
                                                mol=mols[0], nasa_coefficients=nasa_coefficients[0])
            else:
                chemkin_data = ""
                for i in range(len(names)):
                    ck = get_chemkin_file(name=names[i], smiles=smiles[i], method=self.method,
                                          mol=mols[i], nasa_coefficients=nasa_coefficients[i])
                    chemkin_data += ck
            return chemkin_data
        else:
            return nasa_coefficients


class Liquid:
    """:class:`Liquid` contains trained models to predict liquid-phase thermodynamic properties.
    Currently available: Boiling point (bp), critical temperature (tc), critical pressure (pc),
    critical volume (vc), octanol-water partitioning (logp), aqueous solubility (logs)."""
    def __init__(self, prop: str, dimension: str, data_location: Union[str, None] = None):
        """
        :param prop: A string with the property to predict
        (currently supported: `bp`, `tc`, `pc`, `vp`, `logp`, `logs`)
        :param dimension: A string with the dimension of chemical information (`2d` or `3d`)
        """

        self.property = prop
        self.dimension = dimension
        self.inputs = TestInputArguments(dimension=self.dimension)
        self.inputs.property = [self.property]
        if data_location is None:
            self.inputs.save_dir = self.inputs.dir + f"/caesar-data/liquid/{self.property}/{self.dimension}"
        else:
            self.inputs.save_dir = data_location + f"/liquid/{self.property}/{self.dimension}"
        self.models, self.scaler = load_models(self.inputs)

    def predict(self, smiles: Union[str, list], xyz: Union[str, list, None] = None) -> pd.DataFrame:
        """
        This function predicts the liquid-phase property for the corresponding SMILES.
        :param xyz: Either a string of the xyz coordinates or a list of xyz coordinate strings.
        :param smiles: Either a SMILES string or a list of SMILES strings.
        :return: Predictions are returned as a Pandas DataFrame
        """

        if self.dimension == "3d" and xyz is None:
            raise ValueError("Parameter xyz cannot be None! "
                             "You have to provide 3D coordinates to make predictions with a 3D model.")

        df_out = check_input(smiles, xyz, self.inputs)

        dl_test = DataLoader(input_pars=self.inputs, transfer=False, test=True, df=df_out)
        dl_test.scaler = self.scaler
        df_pred = test_external_dataset(self.models, self.scaler, self.inputs, dl_test, return_results=True)
        return df_pred


class Safety:
    def __init__(self, data_location: Union[str, None] = None):
        self.property = ["AIT", "FLTL", "FLTU", "FP", "NBP", "SOLP", "MP"]
        self.dimension = "2d"
        self.inputs = TestInputArguments(dimension=self.dimension)
        self.inputs.property = self.property
        if data_location is None:
            self.inputs.save_dir = self.inputs.dir + f"/caesar-data/liquid/safety/2d"
        else:
            self.inputs.save_dir = data_location + f"/caesar-data/liquid/safety/2d"
        self.models, self.scaler = load_models(self.inputs)

    def predict(self, smiles: Union[str, list]) -> pd.DataFrame:
        df_out = check_input(smiles, None, self.inputs)
        dl_test = DataLoader(input_pars=self.inputs, transfer=False, test=True, df=df_out)
        dl_test.scaler = self.scaler
        df_pred = test_external_dataset(self.models, self.scaler, self.inputs, dl_test, return_results=True)
        return df_pred


def control_thermo_input(smiles, dim, xyz, llot, t, models, scaler, inputs, prop):
    if dim == "3d" and xyz is None:
        raise ValueError("Parameter xyz cannot be None! "
                         "You have to provide 3D coordinates to make predictions with a 3D model.")

    elif dim == "3d" and t != 298.15 and llot is None and prop in ["h", "g"]:
        raise ValueError("Please provide low level-of-theory Dh298 estimate "
                         "to calculate enthalpies of formation at temperatures other than 298.15 K.")

    if type(smiles) is str:
        df_out = read_csv(inputs, smiles=[smiles], xyz_list=[xyz])
        llot = [llot]
    elif type(smiles) is list:
        df_out = read_csv(inputs, smiles=smiles, xyz_list=xyz)
    else:
        raise IndexError(f"Type {type(smiles)} is not supported for parameter smiles.")

    dl_test = DataLoader(input_pars=inputs, transfer=False, test=True, df=df_out)
    dl_test.scaler = scaler
    df_pred = test_external_dataset(models, scaler, inputs, dl_test, return_results=True)

    return df_pred, dl_test, llot


def get_heat_capacity_values(inputs):
    temperatures = np.array([298.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15,
                             363.15, 373.15, 383.15, 393.15, 403.15, 413.15, 423.15,
                             448.15, 473.15, 498.15, 523.15, 548.15, 573.15, 598.15,
                             623.15, 648.15, 673.15, 698.15, 723.15, 748.15, 773.15,
                             798.15, 823.15, 848.15, 873.15, 923.15, 973.15, 1023.15,
                             1073.15, 1123.15, 1173.15, 1223.15, 1273.15, 1323.15, 1373.15,
                             1423.15, 1473.15, 1523.15])
    cp_keys = [f"{key}_prediction" for key in inputs.property[2:]]
    return temperatures, cp_keys


def check_input(smiles, xyz, inputs):
    if type(smiles) is str:
        df_out = read_csv(inputs, smiles=[smiles], xyz_list=xyz)
    elif type(smiles) is list:
        df_out = read_csv(inputs, smiles=smiles, xyz_list=xyz)
    else:
        raise IndexError(f"Type {type(smiles)} is not supported for parameter smiles.")

    return df_out
