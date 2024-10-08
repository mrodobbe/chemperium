import pandas as pd
from chemperium.features.calc_features import periodic_table
from chemperium.data.parse_csv import df_from_csv
from chemperium.data.load_test_data import TestInputArguments
from chemperium.inp import InputArguments
from chemperium.molecule.graph import Mol3DGraph
from chemperium.molecule.batch import featurize_graphs
from chemperium.gaussian.feature_vector import Featurizer, MolFeatureVector
from chemperium.gaussian.histogram import Histograms, Gaussian
from chemperium.features.fingerprint import RDF, Morgan, MACCS
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D
from sklearn.preprocessing import MinMaxScaler
from typing import Union, Tuple, List
import os
import pickle
import numpy as np
import numpy.typing as npt
import tensorflow as tf


class DataLoader:
    """
    A DataLoader object contains all information about a database:
    * SMILES
    * Properties
    * RDKit molecules
    * Molecular graphs
    * Output scaler
    """
    def __init__(self, input_pars: Union[InputArguments, TestInputArguments],
                 transfer: bool = False, test: bool = False, df: pd.DataFrame = None):
        """

        :param input_pars: Input arguments
        :param transfer: Whether to do transfer learning
        :param test: Check if in test phase
        :param df: Pandas DataFrame with database
        """

        self.inp = input_pars
        self.transfer = transfer
        self.test = test
        if df is None:
            self.df = self.load_data()
        else:
            self.df = df
        self.rdmol_list = np.array(self.get_rdmol())
        self.smiles = self.get_smiles()
        self.scaler = self.get_scaler()

        if self.inp.fingerprint is None:
            self.graphs = self.get_graphs()
            self.x = self.get_xs()
        else:
            self.graphs = []
            self.x = self.get_fingerprints()

        self.y = self.get_outputs(inputs=self.inp)

    def load_data(self) -> pd.DataFrame:
        """
        Convert a csv file into a Pandas DataFrame
        :return: Input data in Pandas DataFrame
        """

        if self.transfer:
            df = df_from_csv(self.inp.transfer_file, self.inp.include_3d, self.inp.ff_3d)
        elif self.test:
            df = df_from_csv(self.inp.test_file, self.inp.include_3d, self.inp.ff_3d)
        else:
            df = df_from_csv(self.inp.input_file, self.inp.include_3d, self.inp.ff_3d)
        print(f"We have loaded a database with length {len(df.index)}!")
        return df

    def get_rdmol(self) -> List[Mol]:
        """
        Create RDKit objects for all molecules in the database.
        :return: List with RDKit molecules
        """

        if self.inp.ff_3d:
            mols = []
            for i in self.df.index:
                smi = self.df["smiles"][i]
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    print(f"{smi} cannot be parsed!")
                if not self.inp.no_hydrogens:
                    m = Chem.AddHs(m)
                if m is None:
                    print(f"{smi} cannot be parsed!")
                try:
                    AllChem.EmbedMolecule(m, randomSeed=0xf00d)
                    AllChem.MMFFOptimizeMolecule(m)
                    m.GetConformer(0).GetPositions()
                    mols.append(m)
                except ValueError:
                    self.df = self.df.drop(i)

            return mols
        elif self.inp.no_hydrogens:
            mols = []
            pt = periodic_table()
            inv_pt = {v: k for k, v in pt.items()}
            for i in self.df.index:
                xyz_lines = self.df["xyz"][i].split("\n")[2:]
                xyz_lines = [line.split(" ") for line in xyz_lines]
                clean_lines = []
                for j in xyz_lines:
                    g = [x for x in j if x]
                    clean_lines.append(g)
                coords = [x[1:] for x in clean_lines]
                coords = np.asarray(coords).astype(np.float64)
                atoms = [x[0] for x in clean_lines]
                m = Chem.MolFromSmiles(self.df["smiles"][i])
                try:
                    AllChem.EmbedMolecule(m)
                    c1 = m.GetConformer()
                    for k in range(m.GetNumAtoms()):
                        x, y, z = coords[k]
                        c1.SetAtomPosition(k, Point3D(x, y, z))
                        atom = m.GetAtomWithIdx(k)
                        an = inv_pt.get(atoms[k])
                        atom.SetAtomicNum(an)
                    mols.append(m)
                except ValueError:
                    self.df = self.df.drop(i)
                except IndexError:
                    self.df = self.df.drop(i)

            return mols
        elif not self.inp.include_3d:
            mols = []
            if "isosmiles" in self.df.keys():
                key_value = "isosmiles"
                self.df["smiles"] = self.df["isosmiles"].tolist()
            else:
                key_value = "smiles"
            for i in self.df.index:
                mol = Chem.MolFromSmiles(self.df[key_value][i])

                if mol is None:
                    self.df = self.df.drop(i)
                elif not self.inp.no_hydrogens:
                    mol = Chem.AddHs(mol)
                else:
                    pass

                if mol is not None:
                    if mol.GetNumBonds() < 75:
                        mols.append(mol)
                    else:
                        self.df = self.df.drop(i)
                else:
                    self.df = self.df.drop(i)
            self.df["RDMol"] = mols
            return mols
        else:
            return self.df["RDMol"].to_list()

    def get_smiles(self) -> npt.NDArray[np.str_]:
        """
        Read all SMILES in database.
        :return: A NumPy array with SMILES
        """
        try:
            return self.df["smiles"].to_numpy()
        except KeyError:
            if "isosmiles" in list(self.df.keys()):
                self.df = self.df.rename(columns={"isosmiles": "smiles"})
                return self.df["smiles"].to_numpy()
            elif "SMILES" in list(self.df.keys()):
                self.df = self.df.rename(columns={"SMILES": "smiles"})
                return self.df["smiles"].to_numpy()
            else:
                raise KeyError("No SMILES column detected!")

    def get_graphs(self) -> List[Mol3DGraph]:
        """
        Convert all RDKit molecules in the database to 3D molecular graphs.
        :return: List with Mol3DGraph objects.
        """
        idx = self.df.index
        mgs = np.empty(len(idx), dtype="object")
        for i in reversed(range(len(idx))):
            try:
                graph = Mol3DGraph(self.rdmol_list[i], self.smiles[i], self.inp)
                try:
                    mgs[i] = graph
                except AttributeError as e:
                    print(e)
                    mgs = np.delete(mgs, i)
                    self.df = self.df.drop(idx[i])
                    self.smiles = self.df["smiles"].to_numpy()
                    self.rdmol_list = self.df["RDMol"].to_numpy()
            except (IndexError, NameError) as err:
                print(err)
                mgs = np.delete(mgs, i)
                self.df = self.df.drop(idx[i])
                self.smiles = self.df["smiles"].to_numpy()
                self.rdmol_list = self.df["RDMol"].to_numpy()
        return list(mgs)

    def get_scaler(self) -> MinMaxScaler:
        """
        Create a new 3D scaler object.
        :return: A scikit-learn MinMaxScaler
        """
        return MinMaxScaler(copy=False)

    def get_xs(self) -> Tuple[tf.RaggedTensor, tf.RaggedTensor,
                              tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor,
                              tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
        """
        Convert the Mol3DGraphs into TensorFlow objects.
        :return: Tuple with RaggedTensors
        """
        return featurize_graphs(self.graphs)

    def get_outputs(self,
                    inputs: Union[None, InputArguments, TestInputArguments] = None) -> npt.NDArray[np.float64]:
        """
        Retrieve the output properties from the Pandas DataFrame
        :param inputs: Input arguments
        :return: NumPy array with all properties
        """
        if self.test:
            return np.ones(np.array(self.graphs).shape[0])
        if inputs is not None:
            self.inp = inputs
        if self.inp.save_dl:
            self.df = self.load_data()
        if self.inp.property == "sigma":
            sigma_objs = self.df["sigma"].to_list()
            sigma_values = np.array([sig.sigma for sig in sigma_objs])
            out = sigma_values
        else:
            if self.transfer:
                out = self.df[self.inp.transfer_property].to_numpy()
            else:
                out = self.df[self.inp.property].to_numpy()

        if self.inp.scaler is True:
            self.scaler = self.get_scaler()
            self.scaler.fit(out)
            outputs = self.scaler.transform(out)
            if self.inp.store_models:
                with open(f"{self.inp.save_dir}/scaler.pkl", "wb") as f:
                    pickle.dump(self.scaler, f)
        else:
            outputs = out

        return outputs

    def get_fingerprints(self):
        if self.inp.ff_3d:
            import_type = "smiles"
        else:
            import_type = "precalculated"

        fingerprints = []
        if self.inp.fingerprint == "hdad":
            features = Featurizer(self.rdmol_list, import_type, self.inp)
            try:
                with open(str(self.inp.gmm_file), "rb") as f:
                    gmm_dict = pickle.load(f)
            except:
                hist = Histograms(features.all_features, features.name_all_features, self.inp)
                geometry_dict = hist.histogram_dict
                gauss = Gaussian(self.inp)
                gauss.cluster(geometry_dict)
                gmm_dict = gauss.gmm_dict
            for molecule in features.molecules:
                fp = MolFeatureVector(molecule, gmm_dict, self.inp).vector
                fingerprints.append(fp)
        else:
            for molecule in self.rdmol_list:
                if self.inp.fingerprint == "rdf":
                    fp = RDF(molecule).make_fingerprint(add_mfd=self.inp.mfd)
                elif self.inp.fingerprint == "maccs":
                    fp = MACCS(molecule).make_fingerprint()
                elif self.inp.fingerprint == "morgan":
                    fp = Morgan(molecule).make_fingerprint()
                else:
                    raise KeyError(f"Invalid fingerprint: {self.inp.fingerprint}! Choose from rdf, maccs, morgan, hdad")
                fingerprints.append(fp)

        fingerprints = np.array(fingerprints).astype(np.float32)
        return fingerprints


def input_checker(save_dir: str, gaul: bool = False) -> None:
    """
    Evaluate if save_dir exists
    :param save_dir: Folder to store all data of the training.
    :param gaul: Whether to use GauL-HDAD so that gmm and hist folders are created.
    :return: Function does not return anything
    """
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        print("Folder already exists. Data in this folder will be overwritten.")

    if gaul:
        try:
            os.mkdir(str(save_dir + "/gmm"))
            print("GMM folder created")
        except FileExistsError:
            print("GMM folder already exists.")
        try:
            os.mkdir(str(save_dir + "/hist"))
            print("Hist folder created")
        except FileExistsError:
            print("Hist folder already exists.")


def split_dataset(num_data: int,
                  seed: int = 120897,
                  split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[npt.NDArray[np.int64],
                                                                                      npt.NDArray[np.int64],
                                                                                      npt.NDArray[np.int64],
                                                                                      npt.NDArray[np.int64]]:
    """
    Split dataset into training-validation-test.
    :param num_data: Number of data in the dataset
    :param seed: Seed to initialize pseudo-random generator
    :param split_ratio: Tuple with training-validation-test ratio. Must sum to 1.
    :return: Tuple with four arrays: training-validation-test indices and model indices (=training+validation)
    """
    np.random.seed(seed)
    s = np.arange(num_data)
    np.random.shuffle(s)

    test_pct = split_ratio[-1]
    test_indices = s[:int(test_pct * num_data)]
    model_indices = s[int(test_pct * num_data):]

    train_pct = split_ratio[0] / (1 - test_pct)
    train_indices = model_indices[:int(train_pct * len(model_indices))]
    validation_indices = model_indices[int(train_pct * len(model_indices)):]

    return train_indices, validation_indices, test_indices, model_indices
