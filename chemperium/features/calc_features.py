import numpy as np
from rdkit import Chem


def one_hot_vector(value, length):
    """
    :param value: An integer between 0 and length-1
    :param length: Number of possible values
    :return: A vector of zeros with 1 on position of value
    """
    vector = np.zeros(length)
    vector[value] = 1

    return vector


def mendeleev(n):
    """
    Returns the chemical symbol for an atomic number.
    """
    periodic_table = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        87: "Fr",
        88: "Ra",
        89: "Ac",
        90: "Th",
        91: "Pa",
        92: "U",
        93: "Np",
        94: "Pu",
        95: "Am",
        96: "Cm",
        97: "Bk",
        98: "Cf",
        99: "Es",
        100: "Fm",
        101: "Md",
        102: "No",
        103: "Lr",
        104: "Rf",
        105: "Db",
        106: "Sg",
        107: "Bh",
        108: "Hs",
        109: "Mt",
        110: "Ds",
        111: "Rg",
        112: "Cn",
        113: "Nh",
        114: "Fl",
        115: "Mc",
        116: "Lv",
        117: "Ts",
        118: "Og"
    }
    return periodic_table.get(n)


def periodic_table():
    periodic_table = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        87: "Fr",
        88: "Ra",
        89: "Ac",
        90: "Th",
        91: "Pa",
        92: "U",
        93: "Np",
        94: "Pu",
        95: "Am",
        96: "Cm",
        97: "Bk",
        98: "Cf",
        99: "Es",
        100: "Fm",
        101: "Md",
        102: "No",
        103: "Lr",
        104: "Rf",
        105: "Db",
        106: "Sg",
        107: "Bh",
        108: "Hs",
        109: "Mt",
        110: "Ds",
        111: "Rg",
        112: "Cn",
        113: "Nh",
        114: "Fl",
        115: "Mc",
        116: "Lv",
        117: "Ts",
        118: "Og"
    }
    return periodic_table


def atomic_feature_vector(n: int):
    """
    The algorithm currently allows molecules with H, B, C, N, O, F, Si, P, S, Cl, Ge, As, Se, Br, Sb, Te, I, Au.
    :param n: Atomic number
    :return: One-hot vector of atomic number
    """
    pos_dict = {
        1: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
        14: 6,
        15: 7,
        16: 8,
        17: 9,
        32: 10,
        33: 11,
        34: 12,
        35: 13,
        51: 14,
        52: 15,
        53: 16,
        79: 17
    }
    vector_pos = pos_dict.get(n)
    len_vector = len(pos_dict)

    return one_hot_vector(vector_pos, len_vector)


def hybridization_vector(s):
    pos_dict = {
        "S": 0,
        "SP": 1,
        "SP2": 2,
        "SP3": 3,
        "SP3D": 4,
        "SP3D2": 5
    }
    s = str(s)
    vector_pos = pos_dict.get(s)
    len_vector = len(pos_dict)

    return one_hot_vector(vector_pos, len_vector)


def bond_type_vector(n):
    pos_dict = {
        1.0: 0,
        1.5: 1,
        2.0: 2,
        3.0: 3
    }
    vector_pos = pos_dict.get(n)
    len_vector = len(pos_dict)

    return one_hot_vector(vector_pos, len_vector)


def h_bonds(i, hbond_dict):
    keys = ['H_intra', 'H_inter', 'acc_intra', 'acc_inter', 'don_intra', 'don_inter']
    vec = np.zeros(len(keys))
    for j in range(len(keys)):
        if i in hbond_dict[keys[j]]:
            vec[j] += 1

    return vec


def get_atomic_cdf(atom: Chem.Atom, mol: Chem.Mol, charges, xyz, nout=100, smth=3200, min_p=-1.8, max_p=4.6):
    p = np.linspace(min_p, max_p, num=nout)
    g = []
    atom_id = atom.GetIdx()

    #     for nb in atom.GetNeighbors():
    for nb in mol.GetAtoms():
        nid = nb.GetIdx()
        d_ij = xyz[atom_id][nid]
        if nid == atom_id:
            continue
        if d_ij > 3:
            continue

        gs = np.array(np.exp(-smth * (1 / d_ij ** 2) * (p - (charges[atom_id] - charges[nid])) ** 2))
        g.append(gs)

    g = np.sum(g, axis=0)

    return p, g


def get_atomic_rdf(atom: Chem.Atom, mol: Chem.Mol, xyz, nout=100, smth=1600, max_r=3, decay_width=2, decay_pos=6):
    r = np.linspace(0.8, max_r, num=nout)
    w = 1.0 - 1.0 / (1.0 + np.exp(-decay_width * (r - decay_pos)))
    atom_mass = atom.GetMass()
    atom_id = atom.GetIdx()
    g = []

    for at in mol.GetAtoms():
        nid = at.GetIdx()
        if nid == atom_id:
            continue
        n_mass = at.GetMass()
        wt = (atom_mass * n_mass) ** 0.5
        gs = np.array(w * wt * np.exp(-smth * (r - xyz[nid][atom_id]) ** 2))
        g.append(gs)

    g = np.sum(g, axis=0)
    g = np.nan_to_num(g)

    return r, g


class Sigma:
    def __init__(self, df_value):
        self.df = df_value
        self.sigma = self.df.loc[:, "-0.022":"0.026"].to_numpy().astype(np.float)