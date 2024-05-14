from src.chemperium.inp import InputArguments
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.rdchem import Mol


def get_atomic_features(atom: Chem.Atom, mol: Chem.Mol, hbond_dict: dict, xyz, input_pars: InputArguments):
    feat_list = np.array([])

    if input_pars.rdf:
        try:
            if input_pars.cutoff is None:
                feat_list = np.append(feat_list,
                                      get_atomic_rdf(atom, mol, xyz, nout=100, smth=1600, max_r=3,
                                                     decay_width=2, decay_pos=6)[1])
            else:
                feat_list = np.append(feat_list,
                                      get_atomic_rdf(atom, mol, xyz, nout=100, smth=1600, max_r=input_pars.cutoff,
                                                     decay_width=2, decay_pos=6)[1])
        except AttributeError:
            feat_list = np.append(feat_list,
                                  get_atomic_rdf(atom, mol, xyz, nout=100, smth=1600, max_r=3,
                                                 decay_width=2, decay_pos=6)[1])

    if input_pars.simple_features:
        # atomic number
        feat_list = np.append(feat_list, atomic_feature_vector(atom.GetAtomicNum()))

        # degree of atom
        feat_list = np.append(feat_list, one_hot_vector(atom.GetDegree(), 7))

        # hybridization: S, SP, SP2, SP3, SP3D, SP3D2
        feat_list = np.append(feat_list, hybridization_vector(atom.GetHybridization()))

        # aromaticity: 0 or 1
        feat_list = np.append(feat_list, np.array([int(atom.GetIsAromatic())]))

        # chiral tag
        feat_list = np.append(feat_list, one_hot_vector(atom.GetChiralTag(), 9))

    return feat_list


def get_bond_features(bond: Chem.Bond):
    feat_list = np.array([])
    feat_list = np.append(feat_list, bond_type_vector(bond.GetBondTypeAsDouble()))
    feat_list = np.append(feat_list, np.array([bond.IsInRing()]))
    feat_list = np.append(feat_list, np.array([int(bond.GetIsConjugated())]))

    return feat_list


def get_molecular_features(mol: Mol, spin: int = None):

    # molecular weight
    mw = Descriptors.MolWt(mol)

    # maximal interatomic distance
    r_max = rdmolops.Get3DDistanceMatrix(mol).max()

    # n_heavy
    nheavy = mol.GetNumHeavyAtoms()

    ha_dict = {"C": 0, "N": 0, "O": 0, "S": 0, "F": 0, "Cl": 0, "Br": 0}
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol in ha_dict:
            ha_dict[atom_symbol] += 1

    # number of ring atoms
    ri = mol.GetRingInfo()
    n_ring = len(list(set([a for r in ri.AtomRings() for a in r])))

    # number of aromatic atoms
    arom = mol.GetAromaticAtoms()
    n_arom = len(arom)

    # spin multiplicity
    if spin is None:
        spin = 0

    feat_list = np.array([mw, r_max, nheavy,
                          ha_dict["C"], ha_dict["N"], ha_dict["O"], ha_dict["S"],
                          ha_dict["F"], ha_dict["Cl"], ha_dict["Br"],
                          n_ring, n_arom, spin])

    return feat_list
