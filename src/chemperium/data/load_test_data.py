import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D
import numpy as np
from chemperium.features.calc_features import periodic_table
from chemperium.inp import InputArguments
from keras.models import load_model, Model
import os
import os.path as path
import pickle
from typing import Union, List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from chemperium.model.mpnn import MessagePassing, Readout, DirectedEdgeMessage, BondInputFeatures


class TestInputArguments:
    def __init__(self, dimension: Union[None, str] = None):
        self.training_type = "test"  # training or test
        self.data_type = "molecules"
        self.transfer = False
        self.locked_transfer = True
        self.test = True
        self.store_models = False
        self.dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), "../../.."))
        self.mean_readout = True
        self.scaler = True
        self.ensemble = True
        self.ff_3d = False
        self.no_hydrogens = False
        self.property = [""]
        self.save_dir = ""
        self.test_file = ""
        self.input_file = ""
        self.transfer_file = ""
        self.store_models = False
        self.transfer_property = [""]
        self.save_dl = False
        self.seed = 210995
        self.include_3d = True
        self.charge = False
        self.rdf = False
        self.cdf = False
        self.mfd = False
        self.simple_features = False
        self.num_layers = 5
        self.hidden_size = 512
        self.hidden_activation = "LeakyReLU"
        self.activation = "linear"
        self.dropout = 0.0
        self.batch_normalization = False
        self.l2 = 0.0
        self.bias = True
        self.max_epochs = 700
        self.patience = 50
        self.transfer_patience = 20
        self.batch_size = 16
        self.transfer_batch = 8
        self.ratio = (0.8, 0.1, 0.1)
        self.init_lr = 1e-3
        self.clipvalue = 0.1
        self.decay_rate = 0.95
        self.decay_steps = 10000
        self.masked = False
        self.cutoff = 2.1
        self.hidden_message = 512
        self.depth = 6
        self.representation_size = 256
        self.outer_folds = 1
        self.gmm_file = self.dir + "/src/chemperium/pickle/gmm_dictionary.pickle"
        self.fingerprint = None

        # Plotting

        self.color_1 = "#0F4C81"  # Pantone Classic Blue
        self.color_2 = "#EA733D"  # Pantone Red 032 C
        self.color_3 = "#D01C1F"  # Pantone Fiery Red
        self.font = "Arial"
        self.font_size = 24

        # GauL-HDAD
        self.distances = True
        self.angles = True
        self.dihedrals = True
        self.tol = 1e-4
        self.max_iter = 100
        self.plot_gmm = False
        self.plot_hist = True
        self.radicals = True
        self.carbenium = False

        if dimension is not None:

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


def read_csv(inp: Union[TestInputArguments, InputArguments],
             smiles: Union[List[str], None] = None,
             xyz_list: Union[List[str], None] = None) -> pd.DataFrame:
    if smiles is None:
        csv = inp.test_file
        df_in = pd.read_csv(csv, index_col=None)
        smiles = df_in[0].tolist()
        if inp.include_3d and xyz_list is not None:
            xyz_list = df_in[1].tolist()
        elif inp.include_3d and xyz_list is None:
            raise ValueError("xyz_list cannot be None!")

    if not inp.include_3d:
        df_out = pd.DataFrame({"smiles": smiles})
        df_out["xyz"] = ""
        return df_out

    pt = periodic_table()
    inv_pt = {v: k for k, v in pt.items()}
    data_dict = {"smiles": [],
                 "xyz": [],
                 "RDMol": []}  # type: Dict[str, List[Union[str, Mol]]]
    for i in range(len(smiles)):
        smi = smiles[i]
        if xyz_list is not None:
            xyz = xyz_list[i]
        else:
            raise ValueError("xyz_list cannot be None!")
        try:
            lines = xyz.split("\n")[2:]
            new_lines = [line.split(" ") for line in lines]
            cleans = []
            for j in new_lines:
                g = [x for x in j if x]
                cleans.append(g)
            coords = [x[1:] for x in cleans]
            coords_array = np.asarray(coords).astype(np.float64)
            assert coords_array.shape[1] == 3
            ats = [x[0] for x in cleans]
            m = Chem.MolFromSmiles(smi)
            m1 = Chem.AddHs(m)
            AllChem.EmbedMolecule(m1)
            c1 = m1.GetConformer()
            for k in range(m1.GetNumAtoms()):
                x, y, z = coords_array[k]
                c1.SetAtomPosition(k, Point3D(x, y, z))
                atom = m1.GetAtomWithIdx(k)
                an = inv_pt.get(ats[k])
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


def load_models(inp: Union[InputArguments, TestInputArguments]) -> Tuple[List[Model], Union[MinMaxScaler, None]]:
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
