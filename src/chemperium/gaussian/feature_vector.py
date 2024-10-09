import numpy as np
from chemperium.gaussian.molecular_geometry import MolecularGraph, GaulMolecule
from chemperium.features.calc_features import num_radicals, carbenium_degree, one_hot_vector, dict_to_vector
from chemperium.inp import InputArguments
from importlib import resources
import pickle
from typing import List
import numpy.typing as npt
from rdkit.Chem import Mol


def get_dad_dict(radius: int = 2):
    with resources.path(
            f"chemperium.pickle", f"dad_topological_dict_{radius}.pkl"
    ) as path:
        with open(path, "rb") as f:
            dad_topological_dict = pickle.load(f)

    return dad_topological_dict


def get_dihedral_dict(radius: int = 2):
    with resources.path(
            f"chemperium.pickle", f"dihedral_topological_dict_{radius}.pkl"
    ) as path:
        with open(path, "rb") as f:
            dihedral_topological_dict = pickle.load(f)

    return dihedral_topological_dict


class MolFeatureVector:
    def __init__(self, mol: GaulMolecule, gmm_dict: dict, inp: InputArguments):
        self.mol = mol
        self.gmm_dict = gmm_dict
        self.inp = inp
        self.vector = None
        self.vector = self.construct_vector()

    def construct_vector(self):
        name_features = self.mol.name_all_features
        features = self.mol.all_features
        for i in range(len(features)):
            if name_features[i] in self.gmm_dict.keys():
                f = GeometricFeatureVector(name_features[i], features[i], self.gmm_dict)
                f.construct_vector()
                if self.vector is None:
                    self.vector = f.vector
                else:
                    self.vector = np.add(self.vector, f.vector)
        self.add_molecular_features()

        return self.vector

    def add_molecular_features(self):
        if self.inp.radicals:
            self.vector = np.append(self.vector, np.array([num_radicals(self.mol.molecule)]))
        if self.inp.carbenium:
            self.vector = np.append(self.vector, one_hot_vector(carbenium_degree(self.mol.molecule), 3))

    def get_vector_length(self):
        return len(self.vector)

    def get_vector(self):
        return self.vector


class GeometricFeatureVector:
    def __init__(self, feature: str, value: float, gmm_dict: dict):
        self.feature = feature
        self.gmm_dict = gmm_dict
        self.value = np.array([[value]])
        self.theta_dict = self.gmm_dict[self.feature]
        self.vector = []
        self.representation_dict = {}
        for key in gmm_dict:
            type_feature = sum(1 for c in key if c.isupper())
            if type_feature == 4:
                self.representation_dict[key] = np.zeros(5).astype(np.float32)
            else:
                self.representation_dict[key] = np.zeros(3).astype(np.float32)

    def construct_vector(self):
        if self.feature in self.gmm_dict:
            gmm = self.gmm_dict[self.feature]
            self.representation_dict[self.feature] = gmm.predict_proba(self.value).reshape(-1).astype(np.float32)

        self.vector = dict_to_vector(self.representation_dict)


class MixtureFeatureVector:
    def __init__(self, smiles_list: list, composition: np.ndarray, representation_dict: dict):
        self.smiles_list = smiles_list
        self.composition = composition
        self.representation_dict = representation_dict
        self.representation_list = [representation_dict[smiles] for smiles in smiles_list]
        self.vector = np.zeros(len(self.representation_list[0]))
        self.vector = self.construct_vector()

    def construct_vector(self):
        for i in range(len(self.composition)):
            comp = self.composition[i]
            vec = self.representation_list[i]
            self.vector = np.add(self.vector, comp*vec)

        return self.vector


class ReactionFeatureVector:
    def __init__(self, smiles_list: list, stoichiometry: list, representation_dict: dict):
        self.smiles_list = smiles_list
        self.stoichiometry = stoichiometry
        self.representation_dict = representation_dict
        self.representation_list = []
        for smiles in self.smiles_list:
            try:
                self.representation_list.append(representation_dict[smiles])
            except KeyError:
                random_representation = representation_dict[list(representation_dict)[0]]
                representation = np.array([0.0 for _ in random_representation])
                self.representation_list.append(representation)
                print(f"\nWARNING! Representation for {smiles} could not be obtained.\n")
        # self.representation_list = [representation_dict[smiles] for smiles in smiles_list]
        self.vector = np.zeros(len(self.representation_list[0]))
        self.vector = self.construct_vector()

    def construct_vector(self):
        for i in range(len(self.stoichiometry)):
            comp = self.stoichiometry[i]
            vec = self.representation_list[i]
            self.vector = np.add(self.vector, comp*vec)

        return self.vector


class TopologicalFeatureVector:
    def __init__(self, mol: MolecularGraph, feature_dict: dict, dad: bool = True):
        self.mol = mol
        self.dad = dad
        self.feature_dict = feature_dict

    def construct_vector(self):
        if self.dad:
            feat_names = self.mol.name_bonds + self.mol.name_angles + self.mol.name_dihedrals
            feat_types = list(
                self.mol.bond_types.numpy()
            ) + list(
                self.mol.angle_types.numpy()
            ) + list(
                self.mol.dihedral_types.numpy()
            )

            for feat_name, feat_type in zip(feat_names, feat_types):
                key = f"{feat_name} - {feat_type}"
                if key in self.feature_dict.keys():
                    self.feature_dict[key] = 1
                elif feat_name in self.feature_dict.keys():
                    self.feature_dict[feat_name] = 1
                else:
                    print(f"{key} was not found!")
        else:
            feat_names = self.mol.name_dihedrals
            feat_types = list(self.mol.dihedral_types.numpy())

            for feat_name, feat_type in zip(feat_names, feat_types):
                key = f"{feat_name} - {feat_type}"
                if key in self.feature_dict.keys():
                    self.feature_dict[key] = 1
                elif feat_name in self.feature_dict.keys():
                    self.feature_dict[feat_name] = 1
                else:
                    print(f"{key} was not found!")

        fingerprint = np.array(list(self.feature_dict.values())).astype(np.int32)

        return fingerprint


def make_topological_fingerprints(smiles_list: List[str], dad: bool = True, radius: int = 2) -> npt.NDArray[np.int32]:
    fp_list = []
    if dad:
        feature_dict = get_dad_dict(radius)
    else:
        feature_dict = get_dihedral_dict(radius)

    for smiles in smiles_list:
        try:
            m = MolecularGraph(smiles)
            m.get_geometric_indices(radius)
            tfp = TopologicalFeatureVector(m, feature_dict.copy(), dad)
            fp = tfp.construct_vector()
            fp_list.append(fp)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error with SMILES {smiles}:", e, "The compound is skipped!")
            continue

    fp_list = np.array(fp_list).astype(np.int32)

    return fp_list


class Featurizer:
    def __init__(self, molecule_list: npt.NDArray[Mol], import_type: str, inp: InputArguments):
        self.molecule_list = molecule_list
        self.all_features = []
        self.name_all_features = []
        self.import_type = import_type
        self.inp = inp
        self.molecules = []
        self.bad_molecules = []
        self.get_all_features()

    def get_all_features(self):
        print("Collecting all molecular geometry features...")
        for molecule in self.molecule_list:
            m = GaulMolecule(molecule, self.import_type, self.inp)
            if self.inp.fingerprint == "hdad":
                m.get_all_features()
                m.get_conformer()
                if not m.bad_molecule:
                    self.all_features += m.all_features
                    self.name_all_features += m.name_all_features
                    self.molecules.append(m)
                else:
                    self.bad_molecules.append(m.get_smiles())
            else:
                if not m.bad_molecule:
                    self.molecules.append(m)
                else:
                    self.bad_molecules.append(m.get_smiles())
