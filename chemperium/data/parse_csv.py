from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem, rdMolTransforms
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Geometry import Point3D
import pandas as pd
import numpy as np
from chemperium.features.calc_features import periodic_table


def make_3d_mol(xyz: str):
    pt = periodic_table()
    inv_pt = {v: k for k, v in pt.items()}
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
        m = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineConnectivity(m)
        m1 = Chem.AddHs(m)
        AllChem.EmbedMolecule(m1)
        c1 = m1.GetConformer()
        for j in range(m1.GetNumAtoms()):
            x, y, z = coords[j]
            c1.SetAtomPosition(j, Point3D(x, y, z))
            atom = m1.GetAtomWithIdx(j)
            an = inv_pt.get(ats[j])
            atom.SetAtomicNum(an)
        return m1
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(e)


def df_from_csv(fname: str, include_3d: bool, ff_3d: bool = False) -> pd.DataFrame:
    df = pd.read_csv(fname).reset_index(drop=True)
    smiles_keys = ["smiles", "SMILES", "isosmiles"]
    smiles_key = None

    for key in smiles_keys:
        if key in list(df.keys()):
            smiles_key = key
            break
    if smiles_key is None:
        raise KeyError("No column with SMILES detected in the DataFrame. Please add a column named smiles.")

    if include_3d and not ff_3d:
        if "xyz" not in list(df.keys()) and not ff_3d:
            raise KeyError("XYZ coordinates not provided!")
        if "RDMol" not in df.keys():
            df["RDMol"] = ""
        for i in df.index:
            mol = make_3d_mol(df["xyz"][i])
            if mol is None:
                print(f"WARNING! Could not parse {df[smiles_key][i]}!")
                df = df.drop(i)
            else:
                df.loc[i, "RDMol"] = mol
    elif "xyz" not in list(df.keys()):
        df["xyz"] = ""

    return df
