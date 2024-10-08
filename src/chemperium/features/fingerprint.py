from rdkit import Chem
from rdkit.Chem import Mol, Atom, rdMolTransforms, MACCSkeys, AllChem, rdFingerprintGenerator
from chemperium.features.featurize import get_molecular_features
import numpy as np
import numpy.typing as npt
from typing import Any


class RDF:
    def __init__(self, molecule: Mol):
        self.molecule = molecule
        self.num_atoms = self.molecule.GetNumAtoms()
        self.conformer = self.molecule.GetConformer()
        self.distance_masses = [[]]

    def get_rdf_distances(self):

        distances = []
        name_distances = []

        for i in range(self.num_atoms):
            atom_1 = self.molecule.GetAtomWithIdx(i)
            element_1 = atom_1.GetSymbol()
            for j in range(i + 1, self.num_atoms):
                atom_2 = self.molecule.GetAtomWithIdx(j)
                element_2 = atom_2.GetSymbol()
                distance = rdMolTransforms.GetBondLength(self.conformer, i, j)
                name_distance = "".join(sorted([element_1, element_2]))
                distances += [distance]
                name_distances += [name_distance]
                self.distance_masses.append([atom_1.GetMass(), atom_2.GetMass()])

        self.distance_masses.pop(0)

        return distances, name_distances

    def make_fingerprint(
            self,
            nout=400,
            smth=1600,
            max_r=8,
            decay_width=2,
            decay_pos=6,
            add_mfd: bool = False
    ) -> npt.NDArray[Any]:

        distances, name_distances = self.get_rdf_distances()
        dist = np.asarray(distances).astype(np.float64)
        mw = self.distance_masses

        r = np.linspace(0.8, max_r, num=nout)
        w = 1.0 - 1.0 / (1.0 + np.exp(-decay_width * (r - decay_pos)))
        g = []

        for i in range(len(dist)):
            wt = (mw[i][0] * mw[i][1]) ** 0.5
            gs = np.array(w * wt * np.exp(-smth * (r - dist[i]) ** 2))
            g.append(gs)
        g = np.asarray(g)
        g = np.sum(g, axis=0)

        if add_mfd:
            mfd = get_molecular_features(self.molecule, None)
            g = np.concatenate([g, mfd])

        return g


class MACCS:
    def __init__(self, molecule: Mol):
        self.molecule = molecule

    def make_fingerprint(self) -> npt.NDArray[Any]:
        return np.array(MACCSkeys.GenMACCSKeys(self.molecule))


class Morgan:
    def __init__(self, molecule: Mol, radius: int = 2, bits: int = 1024):
        self.molecule = molecule
        self.radius = radius
        self.bits = bits

    def make_fingerprint(self) -> npt.NDArray[Any]:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, includeChirality=True, fpSize=self.bits)
        fp = fpgen.GetFingerprint(self.molecule)
        fp = np.array(fp)
        return fp


class SymmetryFunctions:
    def __init__(self, molecule: Mol):
        self.molecule = molecule
        self.num_atoms = self.molecule.GetNumAtoms()
        self.conformer = self.molecule.GetConformer()
        self.distance_masses = [[]]

    def get_rdf_distances(self):

        distances = []
        name_distances = []

        for i in range(self.num_atoms):
            atom_1 = self.molecule.GetAtomWithIdx(i)
            element_1 = atom_1.GetSymbol()
            for j in range(i + 1, self.num_atoms):
                atom_2 = self.molecule.GetAtomWithIdx(j)
                element_2 = atom_2.GetSymbol()
                distance = rdMolTransforms.GetBondLength(self.conformer, i, j)
                name_distance = "".join(sorted([element_1, element_2]))
                distances += [distance]
                name_distances += [name_distance]
                self.distance_masses.append([atom_1.GetMass(), atom_2.GetMass()])

        self.distance_masses.pop(0)

        return distances, name_distances

    def make_fingerprint(self, nout=400, smth=1600, max_r=8) -> npt.NDArray[Any]:
        distances, name_distances = self.get_rdf_distances()
        dist = np.asarray(distances).astype(np.float64)
        mw = self.distance_masses

        r = np.linspace(0.8, max_r, num=nout)
        w = 0.5 * (1 + np.cos(np.pi * r / max_r))
        g = []

        for i in range(len(dist)):
            wt = (mw[i][0] * mw[i][1]) ** 0.5
            gs = np.array(w * wt * np.exp(-smth * (r - dist[i]) ** 2))
            g.append(gs)
        g = np.asarray(g)
        g = np.sum(g, axis=0)

        return g
