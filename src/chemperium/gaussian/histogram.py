from sklearn.mixture import GaussianMixture
from chemperium.gaussian.utils import get_dict
from chemperium.postprocessing.plots import gmm_plot, histogram_plot
from chemperium.inp import InputArguments
import pickle
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from typing import Union
from joblib import Parallel, delayed


def split_histograms(bond: str, bond_length: float) -> str:
    """"
    Divides the CC bond range into five types for better coverage of physically relevant bonds
    """
    # TODO: Automate division process
    if bond == "CC":
        if bond_length < 1.206:
            bond = "C1"
        elif bond_length < 1.325:
            bond = "C2"
        elif bond_length < 1.365:
            bond = "C3"
        elif bond_length < 1.465:
            bond = "C5"
        elif bond_length < 1.8:
            bond = "C6"
        elif bond_length < 2.73:
            bond = "C8"
        elif bond_length < 2.95:
            bond = "C9"
        elif bond_length < 4.0:
            bond = "CX"
        else:
            bond = "CY"
    elif bond == "CO":
        if bond_length < 1.27:
            bond = "O1"
        elif bond_length < 1.40:
            bond = "O2"
        elif bond_length < 2.0:
            bond = "O3"
        elif bond_length < 2.27:
            bond = "O4"
        elif bond_length < 2.67:
            bond = "O5"
        elif bond_length < 3.9:
            bond = "O6"
        else:
            bond = "O7"
    elif bond == "CN":
        if bond_length < 1.2:
            bond = "N1"
        elif bond_length < 1.3:
            bond = "N2"
        elif bond_length < 1.4:
            bond = "N3"
        elif bond_length < 1.8:
            bond = "N4"
        else:
            bond = "N5"

    return bond


class Histograms:
    def __init__(self, values: list, value_names: list, inp: InputArguments):
        self.values = values
        self.value_names = value_names
        self.histogram_dict = get_dict(values, value_names, min_values=10)
        self.inp = inp
        if self.inp.plot_hist:
            self.make_histograms()

    def make_histograms(self):
        print("Plotting histograms.")
        for key in tqdm(self.histogram_dict):
            histogram_plot(np.asarray(self.histogram_dict[key]).astype(np.float32), key, self.inp)
        print(f"Stored all histograms in {self.inp.save_dir}hist.")


class Gaussian:
    def __init__(self, inp: InputArguments):
        self.inp = inp
        self.tol = inp.tol
        self.max_iter = inp.max_iter
        self.ll_dict = {}
        self.gmm_dict = {}
        self.n_jobs = inp.processors

    def gaussian_mixture_model(
            self,
            num_peaks: int,
            values: list,
            means_init: Union[None, npt.NDArray[np.float32]] = None
    ):
        gmm = GaussianMixture(num_peaks, means_init=means_init, tol=self.tol, max_iter=self.max_iter)
        gmm.fit(np.array(values).reshape(-1, 1))

        return gmm

    def run_gmm(self, geometry_dict: dict, key: str):
        uppercase_count = sum(1 for char in key if char.isupper())
        if uppercase_count == 4:
            num_peaks = 5
            means_init = np.array([-180, -120, 0, 120, 180]).reshape(-1, 1)
        else:
            num_peaks = 3
            means_init = None

        values = geometry_dict[key]
        gmm_results = self.gaussian_mixture_model(num_peaks, values, means_init)

        return gmm_results

    def cluster(self, geometry_dict: dict):
        if self.n_jobs > 1:
            gmm_info = Parallel(n_jobs=self.n_jobs)(delayed(self.run_gmm)(geometry_dict, key)
                                                    for key in tqdm(geometry_dict))
        else:
            gmm_info = [self.run_gmm(geometry_dict, key) for key in geometry_dict]

        for i, key in enumerate(geometry_dict):
            self.gmm_dict[key] = gmm_info[i]

        self.visualize(geometry_dict)
        self.save_gmm_dict()

    def visualize(self, geometry_dict: dict):
        if self.inp.plot_gmm:
            print("Plot gaussian mixture models.")
            for key in tqdm(self.gmm_dict, desc="GMM plot"):
                gmm_plot(
                    self.gmm_dict[key],
                    np.asarray(geometry_dict[key]).astype(np.float32),
                    key,
                    self.inp
                )

    def save_gmm_dict(self):
        with open(self.inp.save_dir + "/gmm_dictionary.pickle", "wb") as f:
            pickle.dump(self.gmm_dict, f)
