from rdkit import Chem
from rdkit.Chem import AllChem
from chemperium.gaussian.feature_vector import MolFeatureVector, ReactionFeatureVector
from chemperium.gaussian.molecular_geometry import GaulMolecule
from chemperium.gaussian.utils import get_dict
# from postprocessing.plots import output_plot
from chemperium.inp import InputArguments
from chemperium.features.calc_features import remove_atom_mapping
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class FileReader:
    def __init__(self, file_location: str):
        self.file_location = file_location
        self.input_type = self.file_location.split(".")[-1]
        self.file = self.get_file()

        # Add mixtures and reactions

    def get_file(self):
        try:
            with open(self.file_location, "r") as f:
                read_in = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError("The file {} does not exist.".format(self.file_location))
        if self.input_type == "txt":
            file = [line[:-1].split("\t") for line in read_in]  # TODO: Edit for csv
        elif self.input_type == "csv":
            file = [line[:-1].split(",") for line in read_in]
        else:
            raise TypeError(f"Input file {self.file_location} is not supported. Please use a .txt or .csv input file.")
        print("File is read successfully.")
        return file


class InputFileReader(FileReader):
    def __init__(self, inp: InputArguments):
        self.inp = inp
        self.file_location = inp.input_file
        super().__init__(self.file_location)
        self.data_type = inp.data_type
        self.molecule_list = []
        self.value_list = []
        self.rdkit_list = []
        self.reaction_list = []
        self.molecules_in_reaction = []
        self.stoichiometry_list = []
        self.num_outputs = 0
        self.identifier = ""
        self.read_data()
        # if self.data_type == "molecules" and self.num_outputs == 1:
        #     self.plot_output()

    def read_molecules(self):

        self.molecule_list = []
        self.value_list = []

        for line in self.file:
            self.molecule_list += [line[0]]
            self.num_outputs = len(line) - 1
            if self.num_outputs > 1:
                self.value_list.append(line[1:])
            else:
                self.value_list += [line[1]]

    def read_reactions(self):
        self.reaction_list = []
        self.molecules_in_reaction = []
        self.stoichiometry_list = []
        self.value_list = []
        self.molecule_list = []

        for line in self.file:
            rxn_smarts = line[0]
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            self.reaction_list += [AllChem.ReactionToSmiles(rxn)]
            molecules = []
            stoichiometry = []

            for reactant in rxn.GetReactants():
                smi = Chem.MolToSmiles(remove_atom_mapping(reactant))
                smi = GaulMolecule(Chem.MolFromSmiles(smi), "SMILES", self.inp).get_smiles(True)
                self.molecule_list.append(smi)
                molecules.append(smi)
                stoichiometry.append(1)

            for product in rxn.GetProducts():
                smi = Chem.MolToSmiles(remove_atom_mapping(product))
                smi = GaulMolecule(Chem.MolFromSmiles(smi), "SMILES", self.inp).get_smiles(True)
                self.molecule_list.append(smi)
                molecules.append(smi)
                stoichiometry.append(-1)

            self.molecules_in_reaction.append(molecules)
            self.stoichiometry_list.append(stoichiometry)

            self.num_outputs = len(line) - 1
            if self.num_outputs > 1:
                self.value_list.append(line[1:])
            else:
                self.value_list += [line[1]]

        self.molecule_list = list(set(self.molecule_list))

    def read_data(self):
        if self.data_type == "molecules":
            self.read_molecules()
        elif self.data_type == "reactions":
            self.read_reactions()
        else:
            raise TypeError("Only .txt and .csv files are currently supported.")
        self.import_type()
        self.molecule_converter()

    def import_type(self):
        m = self.molecule_list[0]
        if m.__contains__("InChI"):
            self.identifier = "InChI"
        elif m.endswith(".mol"):
            self.identifier = "precalculated"
        else:
            self.identifier = "SMILES"

    def molecule_converter(self):
        if self.identifier == "InChI":
            self.rdkit_list = [inchi_converter(mol) for mol in self.molecule_list]
        elif self.identifier == "SMILES":
            self.rdkit_list = [smiles_converter(mol) for mol in self.molecule_list]
        elif self.identifier == "precalculated":
            print("Reading in all the .mol files!")
            self.rdkit_list = []
            for i in reversed(range(len(self.molecule_list))):
                mol_conv = mol_file_converter(self.molecule_list[i])
                if mol_conv is None:
                    self.value_list.pop(i)
                    self.molecule_list.pop(i)
                else:
                    self.rdkit_list.append(mol_conv)
            self.rdkit_list = list(reversed(self.rdkit_list))
        else:
            raise TypeError("No data loaded.")

    # def plot_output(self):
    #     output_plot(self.rdkit_list, self.value_list)


class PeakFileReader(FileReader):
    def __init__(self, file_location: str):
        super().__init__(file_location)
        self.peak_dict = {}
        self.read_peak_file()

    def read_peak_file(self):

        peak_list = []
        num_peak_list = []

        for line in self.file:
            peak_list += [line[0]]
            num_peak_list += [line[1]]

        self.peak_dict = get_dict(num_peak_list, peak_list)

        return self.peak_dict


class DatasetLoader:
    def __init__(self, file: InputFileReader, molecules: list, gmm: dict, inp: InputArguments):
        self.inp = inp
        self.molecules = molecules
        self.molecule_list = file.molecule_list
        self.value_list = file.value_list
        self.rdkit_list = file.rdkit_list
        self.gmm = gmm
        self.import_type = file.identifier
        self.representation_dict = {}
        self.molecular_representation_list = self.encode_molecules()
        # self.representation_dict = self.make_dict()
        if inp.data_type == "molecules":
            self.data_list = self.molecule_list
            self.representation_list = self.molecular_representation_list
        elif inp.data_type == "reactions":
            self.reaction_list = file.reaction_list
            self.data_list = self.reaction_list
            self.molecules_in_reaction = file.molecules_in_reaction
            self.stoichiometry_list = file.stoichiometry_list
            self.representation_list = self.encode_reactions()
        else:
            raise TypeError(f"Only molecules and reactions supported, not {inp.data_type}.")

    def encode_molecules(self):
        # representation_list = [MolFeatureVector(mol, self.gmm, self.inp).vector for mol in tqdm(self.molecules,
        #                                                                                         desc="Featurize")]
        representation_list = []
        representation_dict = {}
        for mol in tqdm(self.molecules, desc="Featurize"):
            representation = MolFeatureVector(mol, self.gmm, self.inp).vector
            representation_list.append(representation)
            smi = mol.get_smiles(remove_hs=True)
            representation_dict[smi] = representation

        self.representation_dict = representation_dict

        return representation_list

    def encode_reactions(self):
        representation_list = [ReactionFeatureVector(smiles_list, stoichiometry, self.representation_dict).vector
                               for smiles_list, stoichiometry
                               in tqdm(zip(self.molecules_in_reaction, self.stoichiometry_list), desc="RxnFeaturizer")]

        return representation_list

    def make_dict(self):
        representation_dict = {}

        for molecule, representation in zip(self.molecules, self.molecular_representation_list):
            representation_dict[molecule.get_smiles(remove_hs=True)] = representation

        return representation_dict


class DataPart:
    def __init__(self, molecule_list: np.ndarray, value_list: np.ndarray, representation_list: np.ndarray):
        self.data_list = molecule_list
        self.value_list = value_list
        self.representation_list = representation_list
        self.input_size = self.size_representation()
        self.output_size = self.size_output()
        self.num_data = len(molecule_list)

    def size_representation(self):
        return len(self.representation_list[0])

    def size_output(self):
        if len(self.value_list.shape) == 1:
            return 1
        else:
            return self.value_list.shape[1]

    def shuffle_order(self):
        arr = np.arange(len(self.data_list))
        np.random.shuffle(arr)
        return arr

    def select_batch(self, arr, batch_size, it):
        s = arr[it:it+batch_size]
        return DataPart(self.data_list[s], self.value_list[s], self.representation_list[s])


class Dataset:
    def __init__(self, data: DatasetLoader):
        self.data_list = np.asarray(data.data_list)
        self.value_list = np.asarray(data.value_list).astype(np.float)
        self.representation_list = np.asarray(data.representation_list).astype(np.float)

    def split_data(self, kf: tuple):
        index_training = kf[0]
        index_test = kf[1]

        training_data = DataPart(self.data_list[index_training], self.value_list[index_training],
                                 self.representation_list[index_training])

        test_data = DataPart(self.data_list[index_test], self.value_list[index_test],
                             self.representation_list[index_test])

        return training_data, test_data

    def single_split(self, seed: int = 120897, split_ratio: tuple = (0.8, 0.1, 0.1)):
        num_data = len(self.value_list)
        np.random.seed(seed)
        s = np.arange(num_data)
        np.random.shuffle(s)

        test_pct = split_ratio[-1]
        test_indices = s[:int(test_pct * num_data)]
        model_indices = s[int(test_pct * num_data):]

        train_pct = split_ratio[0] / (1 - test_pct)
        train_indices = model_indices[:int(train_pct * len(model_indices))]
        validation_indices = model_indices[int(train_pct * len(model_indices)):]

        training_data = DataPart(self.data_list[train_indices], self.value_list[train_indices],
                                 self.representation_list[train_indices])

        validation_data = DataPart(self.data_list[validation_indices], self.value_list[validation_indices],
                                   self.representation_list[validation_indices])

        test_data = DataPart(self.data_list[test_indices], self.value_list[test_indices],
                             self.representation_list[test_indices])

        return training_data, validation_data, test_data


class DataCleaner:
    def __init__(self, bad_molecules: list, data: InputFileReader):
        self.bad_molecules = bad_molecules
        self.bad_reactions = []
        self.molecules_in_reaction = data.molecules_in_reaction
        self.reaction_list = data.reaction_list
        self.value_list = data.value_list
        self.stoichiometry_list = data.stoichiometry_list

        if data.data_type == "reactions":
            if len(self.bad_molecules) > 0:
                for mol in self.bad_molecules:
                    for i in reversed(range(len(self.molecules_in_reaction))):
                        if mol in data.molecules_in_reaction[i]:
                            self.bad_reactions.append(self.reaction_list[i])
                            self.molecules_in_reaction.pop(i)
                            self.value_list.pop(i)
                            self.reaction_list.pop(i)
                            self.stoichiometry_list.pop(i)

                data.molecules_in_reaction = self.molecules_in_reaction
                data.reaction_list = self.reaction_list
                data.value_list = self.value_list
                data.stoichiometry_list = self.stoichiometry_list

                with open(data.inp.dir + "/bad_reactions.txt", "w") as f:
                    for bad in self.bad_reactions:
                        f.write(bad + "\n")
                    f.close()

                print(f"All reactions that cannot be parsed are listed in {data.inp.dir}/bad_reactions.txt")

        self.data = data


def smiles_converter(molecule):
    return Chem.MolFromSmiles(molecule)


def inchi_converter(molecule):
    return Chem.MolFromInchi(molecule)


def mol_file_converter(molecule):
    return Chem.MolFromMolFile(molecule, removeHs=False)
