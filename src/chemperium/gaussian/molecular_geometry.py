import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Mol, rdMolTransforms, rdmolops, AllChem
from chemperium.features.featurize import get_simple_atomic_features, get_bond_features
from chemperium.gaussian.histogram import split_histograms
from chemperium.inp import InputArguments
from typing import Union, List, Tuple
from itertools import permutations
import numpy.typing as npt


class MolecularGraph:
    """
    This class makes a graph object from an RDKit Mol object and adds features to atoms and bonds.
    """

    def __init__(self, smiles: str, mol: Union[Mol, None] = None):
        self.smiles = Chem.CanonSmiles(smiles)
        if mol is None:
            self.mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))
        else:
            self.mol = mol
        self.num_atoms = 0
        self.num_bonds = 0
        self.atom_features = []
        self.bond_features = []
        self.atom_names = []

        for atom in self.mol.GetAtoms():
            self.atom_features.append(
                get_simple_atomic_features(
                    atom
                )
            )
            self.atom_names.append(atom.GetSymbol())

        self.atom_features = np.asarray(self.atom_features).astype(np.int32)
        self.num_atoms = len(self.atom_features)
        self.num_atoms_vector = tf.repeat(self.num_atoms, self.num_atoms)
        self.num_atoms_vector = 1 / (self.num_atoms_vector - 1)
        self.num_atoms_vector = self.num_atoms_vector.numpy()
        self.atom_feature_length = len(self.atom_features[0])

        for bond in self.mol.GetBonds():
            self.bond_features.append(get_bond_features(bond))
        self.bond_features = np.asarray(self.bond_features).astype(np.int32)

        self.num_bonds = len(self.bond_features)
        self.bond_feature_length = len(self.bond_features[0])
        self.num_bonds_vector = [j for j in range(self.num_bonds)]

        self.bond_pairs = np.empty((2 * self.num_bonds, 2))
        self.name_bonds = []
        self.name_angles = []
        self.name_dihedrals = []

        for atom_1 in range(self.num_atoms):
            for atom_2 in range(atom_1 + 1, self.num_atoms):
                bond = self.mol.GetBondBetweenAtoms(atom_1, atom_2)
                if bond is None:
                    continue

                b_id = bond.GetIdx()
                b_fw = 2 * b_id
                b_rev = 2 * b_id + 1

                self.bond_pairs[b_fw, 0] = int(atom_1)
                self.bond_pairs[b_fw, 1] = int(atom_2)
                self.bond_pairs[b_rev, 0] = int(atom_2)
                self.bond_pairs[b_rev, 1] = int(atom_1)
                self.bond_pairs = np.asarray(self.bond_pairs).astype(np.int64)

        self.atom_neighbors = []
        for i in range(self.num_atoms):
            bond_pairs = self.bond_pairs[:, 1]
            wh = np.where((tf.cast(bond_pairs, "int32") == i))
            neighboring_bonds = []
            for j in wh[0]:
                modulo = j % 2
                if modulo == 0:
                    neighbor = bond_pairs[j + 1]
                else:
                    neighbor = bond_pairs[j - 1]
                neighboring_bonds.append(neighbor)
            neighboring_bonds = np.asarray(neighboring_bonds).astype(np.int64)
            self.atom_neighbors.append(sorted(neighboring_bonds))

        self.atom_features = tf.constant(self.atom_features)
        self.bond_features = tf.constant(self.bond_features)

        bond_features_as_strings = tf.strings.as_string(self.bond_features)
        joined_bond_strings = tf.strings.reduce_join(bond_features_as_strings, separator="", axis=1)
        num_buckets = 1000000
        bond_hashes = tf.strings.to_hash_bucket_strong(
            joined_bond_strings,
            num_buckets,
            key=[123456789, 987654321]
        )
        self.bond_types = bond_hashes
        self.angle_types = tf.constant([])
        self.dihedral_types = tf.constant([])

        self.bond_pairs = tf.constant(self.bond_pairs)[::2]
        self.atom_neighbors = tf.ragged.constant(self.atom_neighbors)

        self.updated_atom_features = None
        self.updated_bond_features = None
        self.updated_angle_features = None
        self.updated_dihedral_features = None

        self.bond_indices = None
        self.angle_indices = None
        self.dihedral_indices = None

        self.angle_trios = tf.constant([])
        self.dihedral_quartets = tf.constant([])

    def update_atom_features(self, radius: int = 4):
        atom_features = self.atom_features
        initial_atom_features = atom_features
        for _ in range(radius):
            atom_features = tf.gather(atom_features, self.atom_neighbors[:, :], axis=0, batch_dims=0)
            atom_features = tf.reduce_sum(atom_features, axis=1)
        new_atom_features = tf.concat([initial_atom_features, atom_features], 1)
        self.updated_atom_features = new_atom_features

        return new_atom_features

    def update_bond_features(self, radius: int = 4):
        if self.updated_atom_features is None:
            updated_atom_features = self.update_atom_features(radius)
        else:
            updated_atom_features = self.updated_atom_features

        self.name_bonds = []
        for pair in self.bond_pairs.numpy():
            element_1 = self.mol.GetAtomWithIdx(int(pair[0])).GetSymbol()
            element_2 = self.mol.GetAtomWithIdx(int(pair[1])).GetSymbol()
            self.name_bonds.append("".join(sorted([element_1, element_2])))

        pair_features = tf.gather(updated_atom_features, self.bond_pairs[:, :], axis=0, batch_dims=0)
        sum_pair_features = tf.reduce_sum(pair_features, axis=2)
        atom_order = tf.argsort(sum_pair_features, axis=1)
        gathered_pair_features = tf.gather(pair_features, atom_order, axis=1, batch_dims=1)
        concatenated_pair_features = tf.concat(tf.unstack(gathered_pair_features, axis=1), axis=1)
        updated_bond_features = tf.concat([self.bond_features, concatenated_pair_features], 1)

        self.updated_bond_features = updated_bond_features
        updated_bond_features_as_strings = tf.strings.as_string(updated_bond_features)
        joined_updated_bond_strings = tf.strings.reduce_join(updated_bond_features_as_strings, separator="", axis=1)
        num_buckets = 1000000
        bond_hashes = tf.strings.to_hash_bucket_strong(
            joined_updated_bond_strings,
            num_buckets,
            key=[123456789, 987654321]
        )
        self.bond_indices = bond_hashes

    def get_angles(self):

        self.angle_trios = []
        self.name_angles = []
        self.angle_bonds = []
        previous_angles = []

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if i == j:
                    continue
                for k in range(self.num_atoms):
                    if k == i or k == j:
                        continue
                    bond_1 = self.mol.GetBondBetweenAtoms(i, j)
                    bond_2 = self.mol.GetBondBetweenAtoms(i, k)
                    bond_3 = self.mol.GetBondBetweenAtoms(j, k)
                    bonds = []
                    bond_forming = []
                    for bond in [bond_1, bond_2, bond_3]:
                        if bond is None:
                            bonds.append(0)
                        else:
                            bond_forming.append(bond.GetIdx())
                            bonds.append(1)
                    if len(bond_forming) == 3:
                        bond_forming = [bond_1.GetIdx(), bond_3.GetIdx()]

                    bond_forming = sorted(bond_forming)

                    if sum(bonds) < 2:
                        continue

                    combination = sorted([i, j, k])
                    if combination in previous_angles:
                        continue
                    else:
                        self.angle_trios.append(combination)
                        self.angle_bonds.append(bond_forming)

                    previous_angles.append(sorted(combination))

        self.angle_trios = tf.constant(np.array(self.angle_trios).astype(np.int32))
        self.angle_bonds = tf.constant(np.array(self.angle_bonds).astype(np.int32))

        if self.angle_trios.shape[0] == 0:
            self.has_angles = False
        else:
            self.has_angles = True

        return self.angle_trios, self.name_angles

    def get_angle_indices(self, radius: int = 4):
        if self.updated_atom_features is None:
            updated_atom_features = self.update_atom_features(radius)
        else:
            updated_atom_features = self.updated_atom_features

        if not hasattr(self, 'angle_trios') or not hasattr(self, 'has_angles'):
            self.angle_trios, self.name_angles = self.get_angles()

        if self.has_angles:

            self.name_angles = []
            for trio in self.angle_trios.numpy():
                element_1 = self.mol.GetAtomWithIdx(int(trio[0])).GetSymbol()
                element_2 = self.mol.GetAtomWithIdx(int(trio[1])).GetSymbol()
                element_3 = self.mol.GetAtomWithIdx(int(trio[2])).GetSymbol()
                self.name_angles.append("".join(sorted([element_1, element_2, element_3])))

            trio_features = tf.gather(updated_atom_features, self.angle_trios[:, :], axis=0, batch_dims=0)
            sum_trio_features = tf.reduce_sum(trio_features, axis=2)
            atom_order = tf.argsort(sum_trio_features, axis=1)
            gathered_trio_features = tf.gather(trio_features, atom_order, axis=1, batch_dims=1)
            concatenated_trio_features = tf.concat(tf.unstack(gathered_trio_features, axis=1), axis=1)
            updated_angle_features = concatenated_trio_features

            self.updated_angle_features = updated_angle_features

            updated_angle_features_as_strings = tf.strings.as_string(updated_angle_features)
            joined_updated_angle_strings = tf.strings.reduce_join(updated_angle_features_as_strings, separator="",
                                                                  axis=1)
            num_buckets = 1000000
            angle_hashes = tf.strings.to_hash_bucket_strong(
                joined_updated_angle_strings,
                num_buckets,
                key=[123456789, 987654321]
            )
            self.angle_indices = angle_hashes

            angle_bond_types = tf.gather(self.bond_types, self.angle_bonds[:, :], axis=0, batch_dims=0)
            angle_bond_features = tf.gather(self.bond_features, self.angle_bonds[:, :], axis=0, batch_dims=0)
            bond_order = tf.argsort(angle_bond_types, axis=1)
            gathered_angle_bond_features = tf.gather(angle_bond_features, bond_order, axis=1, batch_dims=1)
            concatenated_angle_bond_features = tf.concat(tf.unstack(gathered_angle_bond_features, axis=1), axis=1)
            updated_angle_bond_features = concatenated_angle_bond_features

            updated_angle_bond_features_as_strings = tf.strings.as_string(updated_angle_bond_features)
            joined_updated_angle_bond_strings = tf.strings.reduce_join(updated_angle_bond_features_as_strings,
                                                                       separator="", axis=1)
            num_buckets = 1000000
            angle_bond_hashes = tf.strings.to_hash_bucket_strong(
                joined_updated_angle_bond_strings,
                num_buckets,
                key=[123456789, 987654321]
            )
            self.angle_types = angle_bond_hashes

    def get_dihedrals(self):

        self.dihedral_quartets = []
        self.name_dihedrals = []
        previous_dihedrals = []
        self.dihedral_bonds = []

        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if i == j:
                    continue
                if self.mol.GetBondBetweenAtoms(i, j) is None:
                    continue
                for k in range(self.num_atoms):
                    if i == k or j == k:
                        continue
                    if self.mol.GetBondBetweenAtoms(i, k) is not None:
                        bond_forming_reaction_2 = self.mol.GetBondBetweenAtoms(i, k).GetIdx()
                    elif self.mol.GetBondBetweenAtoms(j, k) is not None:
                        bond_forming_reaction_2 = self.mol.GetBondBetweenAtoms(j, k).GetIdx()
                    else:
                        continue
                    for p in range(self.num_atoms):
                        if i == p or j == p or k == p:
                            continue
                        if self.mol.GetBondBetweenAtoms(i, p) is not None:
                            bond_forming_reaction_3 = self.mol.GetBondBetweenAtoms(i, p).GetIdx()
                        elif self.mol.GetBondBetweenAtoms(j, p) is not None:
                            bond_forming_reaction_3 = self.mol.GetBondBetweenAtoms(j, p).GetIdx()
                        elif self.mol.GetBondBetweenAtoms(k, p) is not None:
                            bond_forming_reaction_3 = self.mol.GetBondBetweenAtoms(k, p).GetIdx()
                        else:
                            continue

                        combination = sorted([i, j, k, p])
                        if combination in previous_dihedrals:
                            continue

                        bond_forming = [
                            self.mol.GetBondBetweenAtoms(i, j).GetIdx(),
                            bond_forming_reaction_2,
                            bond_forming_reaction_3
                        ]

                        self.dihedral_quartets += [sorted([i, j, k, p])]
                        previous_dihedrals.append(combination)
                        self.dihedral_bonds.append(sorted(bond_forming))

        self.dihedral_quartets = tf.constant(np.array(self.dihedral_quartets).astype(np.int32))
        self.dihedral_bonds = tf.constant(np.array(self.dihedral_bonds).astype(np.int32))

        if self.dihedral_quartets.shape[0] == 0:
            self.has_dihedrals = False
        else:
            self.has_dihedrals = True

        return self.dihedral_quartets, self.name_dihedrals

    def get_dihedral_indices(self, radius: int = 4):
        if self.updated_atom_features is None:
            updated_atom_features = self.update_atom_features(radius)
        else:
            updated_atom_features = self.updated_atom_features

        if not hasattr(self, 'dihedral_quartets') or not hasattr(self, 'has_dihedrals'):
            self.dihedral_quartets, self.name_dihedrals = self.get_dihedrals()

        if self.has_dihedrals:

            self.name_dihedrals = []
            for quartet in self.dihedral_quartets.numpy():
                element_1 = self.mol.GetAtomWithIdx(int(quartet[0])).GetSymbol()
                element_2 = self.mol.GetAtomWithIdx(int(quartet[1])).GetSymbol()
                element_3 = self.mol.GetAtomWithIdx(int(quartet[2])).GetSymbol()
                element_4 = self.mol.GetAtomWithIdx(int(quartet[3])).GetSymbol()
                self.name_dihedrals.append("".join(sorted([element_1, element_2, element_3, element_4])))

            quartet_features = tf.gather(updated_atom_features, self.dihedral_quartets[:, :], axis=0, batch_dims=0)
            sum_quartet_features = tf.reduce_sum(quartet_features, axis=2)
            atom_order = tf.argsort(sum_quartet_features, axis=1)
            gathered_quartet_features = tf.gather(quartet_features, atom_order, axis=1, batch_dims=1)
            concatenated_quartet_features = tf.concat(tf.unstack(gathered_quartet_features, axis=1), axis=1)
            updated_dihedral_features = concatenated_quartet_features

            self.updated_dihedral_features = updated_dihedral_features

            updated_dihedral_features_as_strings = tf.strings.as_string(updated_dihedral_features)
            joined_updated_dihedral_strings = tf.strings.reduce_join(updated_dihedral_features_as_strings, separator="",
                                                                     axis=1)
            num_buckets = 1000000
            dihedral_hashes = tf.strings.to_hash_bucket_strong(
                joined_updated_dihedral_strings,
                num_buckets,
                key=[123456789, 987654321]
            )
            self.dihedral_indices = dihedral_hashes

            dihedral_bond_types = tf.gather(self.bond_types, self.dihedral_bonds[:, :], axis=0, batch_dims=0)
            dihedral_bond_features = tf.gather(self.bond_features, self.dihedral_bonds[:, :], axis=0, batch_dims=0)
            bond_order = tf.argsort(dihedral_bond_types, axis=1)
            gathered_dihedral_bond_features = tf.gather(dihedral_bond_features, bond_order, axis=1, batch_dims=1)
            concatenated_dihedral_bond_features = tf.concat(tf.unstack(gathered_dihedral_bond_features, axis=1), axis=1)
            updated_dihedral_bond_features = concatenated_dihedral_bond_features

            updated_dihedral_bond_features_as_strings = tf.strings.as_string(updated_dihedral_bond_features)
            joined_updated_dihedral_bond_strings = tf.strings.reduce_join(updated_dihedral_bond_features_as_strings,
                                                                          separator="", axis=1)
            num_buckets = 1000000
            dihedral_bond_hashes = tf.strings.to_hash_bucket_strong(
                joined_updated_dihedral_bond_strings,
                num_buckets,
                key=[123456789, 987654321]
            )
            self.dihedral_types = dihedral_bond_hashes

    def get_geometric_indices(self, radius: int = 4):
        self.update_bond_features(radius)
        self.get_angle_indices(radius)
        if self.has_angles:
            self.get_dihedral_indices(radius)
        else:
            self.has_dihedrals = False


class MolecularGraph3D(MolecularGraph):
    """
    This class makes a 3D molecular graph object from an RDKit Mol object and adds features to atoms and bonds.
    """

    def __init__(self, xyz: str, radius: int = 6, change_atom_order: bool = True):
        mol = Chem.MolFromMolBlock(xyz, removeHs=False)
        if mol is None:
            raise ValueError("RDKit cannot parse this molecule!")
        if change_atom_order:
            Chem.rdMolTransforms.CanonicalizeMol(mol)
        smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        super().__init__(smiles=smi, mol=mol)
        self.get_geometric_indices(radius)
        self.distance_matrix = rdmolops.Get3DDistanceMatrix(self.mol)
        self.conformer = self.mol.GetConformer()
        if self.has_dihedrals:
            self.bond_lengths, self.angles, self.dihedrals = self.get_geometric_features()
        elif self.has_angles:
            self.bond_lengths, self.angles = self.get_geometric_features()
        else:
            self.bond_lengths = self.get_geometric_features()

    def get_geometric_features(self):
        self.bond_lengths = tf.gather_nd(tf.constant(self.distance_matrix), self.bond_pairs)
        if self.has_angles:
            self.angles = []
            for trio in self.angle_trios.numpy():
                angle_permutations = list(permutations(trio))
                for permutation in angle_permutations:
                    if self.mol.GetBondBetweenAtoms(
                            int(permutation[0]), int(permutation[1])
                    ) is not None and self.mol.GetBondBetweenAtoms(
                        int(permutation[1]), int(permutation[2])
                    ) is not None:
                        corner = rdMolTransforms.GetAngleDeg(
                            self.conformer,
                            int(permutation[0]),
                            int(permutation[1]),
                            int(permutation[2])
                        )
                        self.angles.append(corner)
                        break
            self.angles = tf.constant(self.angles)
            if self.has_dihedrals:
                self.dihedrals = []
                found_dihedrals = []
                for quartet in self.dihedral_quartets.numpy():
                    found = False
                    dihedral_permutations = list(permutations(quartet))
                    for permutation in dihedral_permutations:
                        if self.mol.GetBondBetweenAtoms(
                                int(permutation[0]), int(permutation[1])
                        ) is not None and self.mol.GetBondBetweenAtoms(
                            int(permutation[1]), int(permutation[2])
                        ) is not None and self.mol.GetBondBetweenAtoms(
                            int(permutation[2]), int(permutation[3])
                        ) is not None:
                            self.dihedrals.append(rdMolTransforms.GetDihedralDeg(
                                self.conformer,
                                int(permutation[0]),
                                int(permutation[1]),
                                int(permutation[2]),
                                int(permutation[3])
                            ))
                            found = True
                            break
                        elif self.mol.GetBondBetweenAtoms(
                                int(permutation[0]), int(permutation[1])
                        ) is not None and self.mol.GetBondBetweenAtoms(
                            int(permutation[0]), int(permutation[2])
                        ) is not None and self.mol.GetBondBetweenAtoms(
                            int(permutation[0]), int(permutation[3])
                        ) is not None:
                            self.dihedrals.append(rdMolTransforms.GetDihedralDeg(
                                self.conformer,
                                int(permutation[0]),
                                int(permutation[1]),
                                int(permutation[2]),
                                int(permutation[3])
                            ))
                            found = True
                            break
                    found_dihedrals.append(found)
                found_dihedrals = np.array(found_dihedrals)
                self.dihedral_quartets = tf.constant(self.dihedral_quartets.numpy()[found_dihedrals])
                self.name_dihedrals = tf.constant(np.array(self.name_dihedrals)[found_dihedrals])
                self.dihedral_types = tf.constant(np.array(self.dihedral_types)[found_dihedrals])
                self.dihedral_indices = tf.constant(np.array(self.dihedral_indices)[found_dihedrals])
                self.dihedrals = tf.constant(self.dihedrals)

                return self.bond_lengths, self.angles, self.dihedrals
            else:
                return self.bond_lengths, self.angles
        else:
            return self.bond_lengths


def reorder_atoms(mol: Mol, new_order: list, radius: int):
    m = MolecularGraph3D(Chem.MolToMolBlock(Chem.RenumberAtoms(mol, new_order)), radius, False)
    dihedral_quartets = m.dihedral_quartets.numpy()
    last_column = dihedral_quartets[:, -1]
    second_last_column = dihedral_quartets[:, -2]
    second_column = dihedral_quartets[:, 1]
    atom_neighbors = m.atom_neighbors.numpy()

    return m, dihedral_quartets, last_column, second_last_column, second_column, atom_neighbors


def canonicalize_3d_molecule(m: MolecularGraph3D) -> MolecularGraph3D:
    if m.num_atoms < 4:
        return m
    dihedral_quartets = m.dihedral_quartets.numpy()

    if len(dihedral_quartets) == 0:
        return m

    last_column = dihedral_quartets[:, -1]
    second_last_column = dihedral_quartets[:, -2]
    second_column = dihedral_quartets[:, 1]
    atom_neighbors = m.atom_neighbors.numpy()

    if dihedral_quartets[0][1] != 1:
        new_order = [i for i in range(m.num_atoms)]
        new_order[1] = int(dihedral_quartets[0][1])
        new_order[int(dihedral_quartets[0][1])] = 1
        m, dihedral_quartets, last_column, second_last_column, second_column, atom_neighbors = reorder_atoms(
            m.mol,
            new_order,
            2
        )

    if dihedral_quartets[0][2] != 2:
        new_order = [i for i in range(m.num_atoms)]
        new_order[2] = int(dihedral_quartets[0][2])
        new_order[int(dihedral_quartets[0][2])] = 2
        m, dihedral_quartets, last_column, second_last_column, second_column, atom_neighbors = reorder_atoms(
            m.mol,
            new_order,
            2
        )

    if dihedral_quartets[0][3] != 3:
        new_order = [i for i in range(m.num_atoms)]
        new_order[3] = int(dihedral_quartets[0][3])
        new_order[int(dihedral_quartets[0][3])] = 3
        m, dihedral_quartets, last_column, second_last_column, second_column, atom_neighbors = reorder_atoms(
            m.mol,
            new_order,
            2
        )

    i = 3  # Starting atom index

    already_replaced = []
    last_visited = [-1]
    problematic_atoms = [-1]

    while i < m.num_atoms:
        if i in last_column:  # Check if i is in the last column
            i += 1
            continue
        else:
            if last_visited[-1] == i:
                already_replaced = [k for k in range(i)]
            else:
                last_visited.append(i)
            smaller_neighbors = np.where(atom_neighbors[i] < i)[0]
            found = False
            if len(smaller_neighbors) == 0 or i in already_replaced or i in problematic_atoms:
                for j in range(len(dihedral_quartets)):
                    last_element = dihedral_quartets[j][-1]
                    if last_element > i and last_element not in already_replaced:
                        replace_with = last_element
                        if i in problematic_atoms:
                            problematic_atoms = [-1]
                        found = True
                        break
            else:
                largest_nb = max(atom_neighbors[i][np.where(atom_neighbors[i] < i)[0]])
                try:
                    replace_with = swap_atoms(i, largest_nb, dihedral_quartets)
                    found = True
                except ValueError:
                    problematic_atoms.append(i)
                    i -= 1
                    continue

            if not found:
                if i in second_last_column:
                    chosen_dihedral = int(np.where(second_last_column == i)[0][0])
                    replace_with = dihedral_quartets[chosen_dihedral][-1]
                elif i in second_column:
                    chosen_dihedral = int(np.where(second_column == i)[0][0])
                    replace_with = dihedral_quartets[chosen_dihedral][2]

            already_replaced.append(replace_with)
            already_replaced.append(i)

            new_order = [i for i in range(m.num_atoms)]
            new_order[int(i)] = int(replace_with)
            new_order[int(replace_with)] = int(i)

            m, dihedral_quartets, last_column, second_last_column, second_column, atom_neighbors = reorder_atoms(
                m.mol,
                new_order,
                2
            )

            i = 3

    if len(np.where(atom_neighbors[1] < 1)[0]) == 0:
        new_order = [i for i in range(m.num_atoms)]
        new_order[1] = int(min(atom_neighbors[0]))
        new_order[int(min(atom_neighbors[0]))] = 1
        m = MolecularGraph3D(Chem.MolToMolBlock(Chem.RenumberAtoms(m.mol, new_order)), 2, False)
        atom_neighbors = m.atom_neighbors.numpy()

    if len(np.where(atom_neighbors[2] < 2)[0]) == 0:
        new_order = [i for i in range(m.num_atoms)]
        new_order[2] = int(min(atom_neighbors[2]))
        new_order[int(min(atom_neighbors[2]))] = 2
        m = MolecularGraph3D(Chem.MolToMolBlock(Chem.RenumberAtoms(m.mol, new_order)), 2, False)

    return m


def swap_atoms(idx: int, nb_idx: int, dihedral_quartets: npt.NDArray[np.int32]) -> int:
    dihedral_ids = np.where(dihedral_quartets[:, 2] == nb_idx)[0]
    if len(dihedral_ids) == 0:
        dihedral_ids = np.where(dihedral_quartets[:, 1] == nb_idx)[0]

    selected_dihedrals = dihedral_quartets[dihedral_ids]
    allowed_dihedrals = np.where(selected_dihedrals[:, -1] > idx)[0]
    retained_dihedral = np.min(dihedral_ids[allowed_dihedrals])
    selected_atom = dihedral_quartets[retained_dihedral][-1]

    return selected_atom


class GaulMolecule:

    def __init__(self, molecule: Chem.rdchem.Mol, import_type: str, inp: InputArguments):
        self.inp = inp
        self.bad_molecule = False
        self.molecule = molecule
        self.conformer = molecule
        #  self.get_conformer()
        self.import_type = import_type
        self.num_atoms = 0
        self.get_num_atoms()
        self.distance_matrix = [[]]
        self.distances = []
        self.name_distances = []
        self.angles = []
        self.name_angles = []
        self.dihedrals = []
        self.name_dihedrals = []
        self.all_features = []
        self.name_all_features = []
        self.distance_masses = [[]]
        self.include_distances = inp.distances
        self.include_angles = inp.angles
        self.include_dihedrals = inp.dihedrals

    def optimize_geometry(self):
        mol = Chem.AddHs(self.molecule)
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        try:
            em = AllChem.EmbedMolecule(mol, params=params)
            if em < 0:
                emr = AllChem.EmbedMolecule(mol, useRandomCoords=True)
                if emr < 0:
                    smi = Chem.MolToSmiles(self.molecule)
                    print(f"Molecule {smi} cannot be embedded.")
                    self.bad_molecule = True
        except RuntimeError:
            self.bad_molecule = True
            smi = Chem.MolToSmiles(self.molecule)
            print(f"Molecule {smi} cannot be embedded.")

        if not self.bad_molecule:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=10000)
            self.molecule = mol
            self.get_conformer()
        return self

    def get_smiles(self, remove_hs=False):
        if remove_hs:
            mol = Chem.RemoveHs(self.molecule)
        else:
            mol = self.molecule
        return Chem.MolToSmiles(mol)

    def get_inchi(self):
        return Chem.MolToInchi(self.molecule)

    def get_conformer(self):
        self.conformer = self.molecule.GetConformer()

    def get_num_atoms(self):
        self.num_atoms = self.molecule.GetNumAtoms()

    def get_distance_matrix(self):
        self.distance_matrix = rdmolops.Get3DDistanceMatrix(self.molecule)

    def get_distances(self):

        self.distances = []
        self.name_distances = []

        for i in range(self.num_atoms):
            atom_1 = self.molecule.GetAtomWithIdx(i)
            element_1 = atom_1.GetSymbol()
            for j in range(i+1, self.num_atoms):
                atom_2 = self.molecule.GetAtomWithIdx(j)
                element_2 = atom_2.GetSymbol()
                distance = self.distance_matrix[i][j]
                name_distance = "".join(sorted([element_1, element_2]))
                name_distance = split_histograms(name_distance, distance)
                self.distances += [distance]
                self.name_distances += [name_distance]

        return self.distances, self.name_distances

    def get_angles(self):

        self.angles = []
        self.name_angles = []
        previous_angles = []

        for i in range(self.num_atoms):
            atom_1 = self.molecule.GetAtomWithIdx(i)
            element_1 = atom_1.GetSymbol()
            for j in range(self.num_atoms):
                if i == j:
                    continue
                atom_2 = self.molecule.GetAtomWithIdx(j)
                element_2 = atom_2.GetSymbol()
                for k in range(self.num_atoms):
                    if k == i or k == j:
                        continue
                    atom_3 = self.molecule.GetAtomWithIdx(k)
                    element_3 = atom_3.GetSymbol()
                    combination = [i, j, k]
                    name_angle = "".join(sorted([element_1, element_2, element_3]))
                    angle_permutations = list(permutations(combination))
                    for permutation in angle_permutations:
                        if list(permutation) in previous_angles:
                            continue

                        if self.molecule.GetBondBetweenAtoms(
                                int(permutation[0]), int(permutation[1])
                        ) is not None and self.molecule.GetBondBetweenAtoms(
                            int(permutation[1]), int(permutation[2])
                        ) is not None:
                            corner = rdMolTransforms.GetAngleDeg(
                                self.conformer,
                                int(permutation[0]),
                                int(permutation[1]),
                                int(permutation[2])
                            )
                            self.angles.append(corner)
                            self.name_angles.append(name_angle)
                            previous_angles.append(list(permutation))
                            previous_angles.append([permutation[2], permutation[1], permutation[0]])

        return self.angles, self.name_angles

    def get_dihedrals(self):

        self.dihedrals = []
        self.name_dihedrals = []
        previous_dihedrals = []

        for i in range(self.num_atoms):
            atom_1 = self.molecule.GetAtomWithIdx(i)
            element_1 = atom_1.GetSymbol()

            for j in range(self.num_atoms):
                if i == j:
                    continue
                if self.molecule.GetBondBetweenAtoms(int(i), int(j)) is None:
                    continue

                atom_2 = self.molecule.GetAtomWithIdx(j)
                element_2 = atom_2.GetSymbol()
                if element_1 == "H" and element_2 == "H":
                    continue

                for k in range(self.num_atoms):
                    if i == k or j == k:
                        continue
                    if self.molecule.GetBondBetweenAtoms(
                            int(i), int(k)
                    ) is None and self.molecule.GetBondBetweenAtoms(
                        int(j), int(k)
                    ) is None:
                        continue
                    atom_3 = self.molecule.GetAtomWithIdx(k)
                    element_3 = atom_3.GetSymbol()

                    for p in range(self.num_atoms):
                        if i == p or j == p or k == p:
                            continue
                        if self.molecule.GetBondBetweenAtoms(
                                int(i), int(p)
                        ) is None and self.molecule.GetBondBetweenAtoms(
                            int(j), int(p)
                        ) is None and self.molecule.GetBondBetweenAtoms(
                            int(k), int(p)
                        ) is None:
                            continue
                        combination = list(sorted([i, j, k, p]))
                        if combination in previous_dihedrals:
                            continue
                        atom_4 = self.molecule.GetAtomWithIdx(p)
                        element_4 = atom_4.GetSymbol()
                        name_dihedral = "".join(sorted([element_1, element_2, element_3, element_4]))

                        dihedral_permutations = list(permutations(combination))
                        for permutation in dihedral_permutations:
                            if self.molecule.GetBondBetweenAtoms(
                                    int(permutation[0]), int(permutation[1])
                            ) is not None and self.molecule.GetBondBetweenAtoms(
                                int(permutation[1]), int(permutation[2])
                            ) is not None and self.molecule.GetBondBetweenAtoms(
                                int(permutation[2]), int(permutation[3])
                            ) is not None:
                                dihedral = rdMolTransforms.GetDihedralDeg(
                                    self.conformer,
                                    int(permutation[0]),
                                    int(permutation[1]),
                                    int(permutation[2]),
                                    int(permutation[3])
                                )
                                if not str(dihedral).__contains__('nan'):
                                    self.dihedrals.append(dihedral)
                                    self.name_dihedrals.append(name_dihedral)
                                    previous_dihedrals.append(combination)
                                    break
                            elif self.molecule.GetBondBetweenAtoms(
                                    int(permutation[0]), int(permutation[1])
                            ) is not None and self.molecule.GetBondBetweenAtoms(
                                int(permutation[0]), int(permutation[2])
                            ) is not None and self.molecule.GetBondBetweenAtoms(
                                int(permutation[0]), int(permutation[3])
                            ) is not None:
                                dihedral = rdMolTransforms.GetDihedralDeg(
                                    self.conformer,
                                    int(permutation[0]),
                                    int(permutation[1]),
                                    int(permutation[2]),
                                    int(permutation[3])
                                )
                                if not str(dihedral).__contains__('nan'):
                                    self.dihedrals.append(dihedral)
                                    self.name_dihedrals.append(name_dihedral)
                                    previous_dihedrals.append(combination)
                                    break

        return self.dihedrals, self.name_dihedrals

    def get_all_features(self):
        if self.import_type != "precalculated":
            m = self.optimize_geometry()
        else:
            m = self

        if not m.bad_molecule:
            m.get_conformer()
            m.get_num_atoms()
            m.get_distance_matrix()

            all_features = []
            name_all_features = []

            if self.include_distances:
                distances, name_distances = m.get_distances()
                all_features += distances
                name_all_features += name_distances

            if self.include_angles:
                angles, name_angles = m.get_angles()
                all_features += angles
                name_all_features += name_angles

            if self.include_dihedrals:
                dihedrals, name_dihedrals = m.get_dihedrals()
                all_features += dihedrals
                name_all_features += name_dihedrals

            self.all_features = all_features
            self.name_all_features = name_all_features
