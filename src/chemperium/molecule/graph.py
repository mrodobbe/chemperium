import tensorflow as tf
from chemperium.features.featurize import *
from chemperium.inp import InputArguments
from chemperium.data.load_test_data import TestInputArguments
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol
import numpy as np
from typing import Union


class Mol3DGraph:
    """
    This class makes a three-dimensional graph object from an RDKit Mol object and adds features to atoms and bonds.
    """

    def __init__(self, mol: Mol, smiles: str, input_pars: Union[InputArguments, TestInputArguments],
                 spin: Union[None, int] = None):
        self.input_pars = input_pars
        if self.input_pars.no_hydrogens:
            try:
                self.mol = Chem.RemoveHs(mol)
            except:
                self.mol = mol
        else:
            self.mol = mol
        self.smiles = smiles
        self.num_atoms = 0
        self.num_bonds = 0
        self.atom_features = []
        self.bond_features = []

        if self.input_pars.include_3d:
            self.xyz = self.mol.GetConformer().GetPositions()
            self.distance_matrix = rdmolops.Get3DDistanceMatrix(self.mol)
        else:
            self.xyz = np.array([[0, 0, 0] for _ in range(self.mol.GetNumAtoms())])
            self.distance_matrix = None

        if self.input_pars.mfd:
            self.mol_features = np.array([get_molecular_features(self.mol, spin=spin)])
        else:
            self.mol_features = np.array([np.zeros(13)])

        for atom in self.mol.GetAtoms():
            self.atom_features.append(
                get_atomic_features(atom, self.mol,
                                    self.distance_matrix, self.input_pars))
        self.atom_features = np.asarray(self.atom_features)
        self.num_atoms = len(self.atom_features)
        self.num_atoms_vector = tf.repeat(self.num_atoms, self.num_atoms)
        self.num_atoms_vector = 1 / (self.num_atoms_vector - 1)
        self.num_atoms_vector = self.num_atoms_vector.numpy()
        self.atom_feature_length = len(self.atom_features[0])

        for bond in self.mol.GetBonds():
            self.bond_features.append(get_bond_features(bond))
        self.bond_features = np.asarray(self.bond_features)
        self.num_bonds = len(self.bond_features)
        self.bond_feature_length = len(self.bond_features[0])

        self.num_bonds_vector = [j for j in range(self.num_bonds)]

        try:
            cutoff_value = self.input_pars.cutoff
        except AttributeError:
            cutoff_value = None

        if cutoff_value is None or self.distance_matrix is None:

            self.bond_representations = np.empty((2 * self.num_bonds,
                                                  self.atom_feature_length + self.bond_feature_length))
            self.bond_pairs = np.empty((2 * self.num_bonds, 2))

            for atom_1 in range(self.num_atoms):
                for atom_2 in range(atom_1 + 1, self.num_atoms):
                    bond = self.mol.GetBondBetweenAtoms(atom_1, atom_2)
                    if bond is None:
                        continue

                    b_id = bond.GetIdx()
                    b_fw = 2 * b_id
                    b_rev = 2 * b_id + 1

                    self.bond_representations[b_fw, :] = np.concatenate(
                        (self.atom_features[atom_1], self.bond_features[b_id]))
                    self.bond_representations[b_rev, :] = np.concatenate(
                        (self.atom_features[atom_2], self.bond_features[b_id]))
                    self.bond_pairs[b_fw, 0] = int(atom_1)
                    self.bond_pairs[b_fw, 1] = int(atom_2)
                    self.bond_pairs[b_rev, 0] = int(atom_2)
                    self.bond_pairs[b_rev, 1] = int(atom_1)
                self.bond_pairs = np.asarray(self.bond_pairs).astype(np.int64)

        else:
            self.num_bonds = int(((np.asarray(self.distance_matrix) <
                                   self.input_pars.cutoff) & (np.asarray(self.distance_matrix) > 0.2)).sum() / 2)
            self.bond_representations = np.empty((2 * self.num_bonds,
                                                  self.atom_feature_length + self.bond_feature_length + 1))
            self.bond_pairs = np.empty((2 * self.num_bonds, 2))
            if len(self.bond_pairs) > 250:
                print(f"The given molecule {smiles} is too big and cannot be parsed. "
                      f"The maximal number of bond pairs (inside the sphere) is set "
                      f"at 250 and {smiles} has {len(self.bond_pairs)} pairs.")
                raise IndexError("Molecule is too big.")

            self.bond_count = 0
            for atom_1 in range(self.num_atoms):
                for atom_2 in range(atom_1 + 1, self.num_atoms):
                    if self.distance_matrix[atom_1][atom_2] > self.input_pars.cutoff:
                        continue
                    elif self.distance_matrix[atom_1][atom_2] < 0.1:
                        continue
                    else:
                        bond = self.mol.GetBondBetweenAtoms(atom_1, atom_2)
                        b_fw = 2 * self.bond_count
                        b_rev = 2 * self.bond_count + 1
                        if bond is None:
                            self.bond_representations[b_fw, :] = np.concatenate(
                                (self.atom_features[atom_1], np.zeros(self.bond_feature_length),
                                 np.array([self.distance_matrix[atom_1][atom_2]])))
                            self.bond_representations[b_rev, :] = np.concatenate(
                                (self.atom_features[atom_2], np.zeros(self.bond_feature_length),
                                 np.array([self.distance_matrix[atom_1][atom_2]])))
                        else:
                            b_id = bond.GetIdx()
                            self.bond_representations[b_fw, :] = np.concatenate(
                                (self.atom_features[atom_1], self.bond_features[b_id],
                                 np.array([self.distance_matrix[atom_1][atom_2]])))
                            self.bond_representations[b_rev, :] = np.concatenate(
                                (self.atom_features[atom_2], self.bond_features[b_id],
                                 np.array([self.distance_matrix[atom_1][atom_2]])))
                        self.bond_pairs[b_fw, 0] = int(atom_1)
                        self.bond_pairs[b_fw, 1] = int(atom_2)
                        self.bond_pairs[b_rev, 0] = int(atom_2)
                        self.bond_pairs[b_rev, 1] = int(atom_1)
                        self.bond_count += 1

            self.bond_pairs = np.asarray(self.bond_pairs).astype(np.int64)
            br = []
            bp = []
            for i in range(self.bond_representations.shape[0]):
                if np.sum(self.bond_representations[i]) > 1e-50:
                    br.append(self.bond_representations[i])
                    bp.append(self.bond_pairs[i])
            self.bond_representations = np.array(br)
            self.bond_pairs = np.array(bp)

        self.bond_neighbors = []
        for i in range(len(self.bond_pairs)):
            nbs = np.where(self.bond_pairs[:, 1] == self.bond_pairs[i][0])[0]
            new_nbs = np.setdiff1d(nbs, np.where(self.bond_pairs[:, 0] == self.bond_pairs[i][1])[0])
            self.bond_neighbors.append(new_nbs)

        self.atom_neighbors = []
        self.atom_bond_neighbors = []
        for i in range(self.num_atoms):
            bond_pairs = self.bond_pairs[:, 1]
            wh = np.where((tf.cast(bond_pairs, "int32") == i))
            self.atom_bond_neighbors.append(wh[0])
            neighboring_bonds = []
            for j in wh[0]:
                modulo = j % 2
                if modulo == 0:
                    neighbor = bond_pairs[j + 1]
                else:
                    neighbor = bond_pairs[j - 1]
                neighboring_bonds.append(neighbor)
            neighboring_bonds = np.asarray(neighboring_bonds).astype(np.int64)
            self.atom_neighbors.append(neighboring_bonds)
