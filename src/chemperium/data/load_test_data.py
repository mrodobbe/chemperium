import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
from chemperium.features.calc_features import periodic_table
from chemperium.inp import InputArguments
from tensorflow.keras.models import load_model
import os
import os.path as path
import pickle
from typing import Union
from chemperium.model.mpnn import MessagePassing, Readout, DirectedEdgeMessage, BondInputFeatures


class TestInputArguments:
    def __init__(self, dimension: str):
        self.training_type = "test"  # training or test
        self.transfer = False
        self.test = True
        self.dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), "../../.."))
        self.mean_readout = True
        self.scaler = True
        self.ensemble = True
        self.ff_3d = False
        self.no_hydrogens = False
        self.property = [""]
        self.save_dir = ""

        if dimension == "2d":
            self.include_3d = False
            self.simple_features = True
            self.mean_readout = True
            self.rdf = False
            self.cdf = False
            self.scaler = True
            self.ensemble = True
            self.ff_3d = False
            self.no_hydrogens = False
            self.gasteiger = False
            self.charge = False
            self.property = [""]
            self.save_dir = ""
            self.mfd = False

        else:
            self.include_3d = True
            self.simple_features = False
            self.rdf = True
            self.mfd = True
            self.cutoff = 2.1


def read_csv(inp: Union[TestInputArguments, InputArguments], smiles: list = None, xyz_list: list = None):
    if smiles is None:
        csv = inp.test_csv
        df_in = pd.read_csv(csv, index_col=None)
        smiles = df_in[0].tolist()
        if inp.include_3d:
            xyz_list = df_in[1].tolist()

    if not inp.include_3d:
        df_out = pd.DataFrame({"smiles": smiles})
        df_out["xyz"] = ""
        return df_out

    pt = periodic_table()
    inv_pt = {v: k for k, v in pt.items()}
    data_dict = {"smiles": [], "xyz": [], "RDMol": []}
    for i in range(len(smiles)):
        smi = smiles[i]
        xyz = xyz_list[i]
        try:
            lines = xyz.split("\n")[2:]
            new_lines = [line.split(" ") for line in lines]
            cleans = []
            for j in new_lines:
                g = [x for x in j if x]
                cleans.append(g)
            coords = [x[1:] for x in cleans]
            coords = np.asarray(coords).astype(np.float64)
            ats = [x[0] for x in cleans]
            m = Chem.MolFromSmiles(smi)
            m1 = Chem.AddHs(m)
            AllChem.EmbedMolecule(m1)
            c1 = m1.GetConformer()
            for j in range(m1.GetNumAtoms()):
                x, y, z = coords[j]
                c1.SetAtomPosition(j, Point3D(x, y, z))
                atom = m1.GetAtomWithIdx(j)
                an = inv_pt.get(ats[j])
                atom.SetAtomicNum(an)
            data_dict["smiles"].append(smi)
            data_dict["xyz"].append(xyz)
            data_dict["RDMol"].append(m1)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            print(f"Molecule {smiles[i]} could not be parsed.")

    df_out = pd.DataFrame(data_dict)

    return df_out


def load_models(inp: Union[InputArguments, TestInputArguments]):
    # models = [load_model(f"{inp.save_dir}/{model}",
    #                      custom_objects={'MessagePassing': MessagePassing, 'Readout': Readout}, compile=False)
    #           for model in os.listdir(f"{inp.save_dir}") if model.endswith(".h5")]
    models = []
    for model in os.listdir(f"{inp.save_dir}"):
        if model.endswith(".keras") or model.endswith(".h5"):
            keras_model = load_model(f"{inp.save_dir}/{model}",
                                     custom_objects={'MessagePassing': MessagePassing,
                                                     'Readout': Readout,
                                                     'BondInputFeatures': BondInputFeatures,
                                                     'DirectedEdgeMessage': DirectedEdgeMessage},
                                     compile=False)
            models.append(keras_model)

    if inp.scaler:
        with open(f"{inp.save_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None

    return models, scaler
