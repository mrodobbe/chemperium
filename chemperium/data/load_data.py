from chemperium.inp import InputArguments
import pandas as pd
from chemperium.molecule.batch import *
from chemperium.features.calc_features import periodic_table
from chemperium.data.parse_csv import df_from_csv
from chemperium.data.load_test_data import TestInputArguments
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from sklearn.preprocessing import MinMaxScaler
from typing import Union
import os
import pickle


class DataLoader:
    def __init__(self, input_pars: Union[InputArguments, TestInputArguments],
                 transfer: bool = False, test: bool = False, df: pd.DataFrame = None):

        self.inp = input_pars
        self.transfer = transfer
        self.test = test
        if df is None:
            self.df = self.load_data()
        else:
            self.df = df
        self.rdmol_list = self.get_rdmol()
        self.smiles = self.get_smiles()
        self.charges = self.get_charges()
        self.graphs = self.get_graphs()
        self.scaler = self.get_scaler()
        self.x = self.get_xs()
        self.y = self.get_outputs(inputs=self.inp)

    def load_data(self):
        if self.transfer:
            df = df_from_csv(self.inp.transfer_file, self.inp.include_3d, self.inp.ff_3d)
        elif self.test:
            df = df_from_csv(self.inp.test_file, self.inp.include_3d, self.inp.ff_3d)
        else:
            df = df_from_csv(self.inp.input_file, self.inp.include_3d, self.inp.ff_3d)
        print(f"We have loaded a database with length {len(df.index)}!")
        return df

    def get_rdmol(self):
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

            mols = np.asarray(mols)
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
            mols = np.asarray(mols)
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
            return np.asarray(mols)
        else:
            return self.df["RDMol"].to_numpy()

    def get_smiles(self):
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

    def get_charges(self):
        if "chargeVector" in self.df.keys():
            charges = [np.asarray(self.df["chargeVector"][idx]) for idx in self.df.index]
            return charges
        else:
            return None

    def get_spin(self):
        if "multiplicity" in self.df.keys():
            spin = self.df["multiplicity"].to_numpy()
            return spin

    def get_graphs(self):
        idx = self.df.index
        mgs = np.empty(len(idx), dtype="object")
        for i in reversed(range(len(idx))):
            try:
                graph = Mol3DGraph(self.rdmol_list[i], self.smiles[i], self.df["xyz"][idx[i]], self.inp)
                try:
                    mgs[i] = graph
                    hb = mgs[i].num_hbonds
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
        return mgs

    def get_scaler(self):
        return MinMaxScaler(copy=False)

    def get_xs(self):
        return featurize_graphs(self.graphs)

    def get_outputs(self, inputs: InputArguments = None):
        if self.test:
            return np.ones(self.graphs.shape[0])
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


def input_checker(save_dir: str):
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        print("Folder already exists. Data in this folder will be overwritten.")


def split_dataset(num_data: int, seed: int = 120897, split_ratio: tuple = (0.8, 0.1, 0.1)):
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
