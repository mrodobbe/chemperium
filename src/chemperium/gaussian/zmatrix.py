from chemperium.gaussian.molecular_geometry import MolecularGraph3D, canonicalize_3d_molecule
import numpy as np
import tensorflow as tf
from ase.io.gaussian import _read_zmatrix
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom, Mol


def create_z_matrix(m: MolecularGraph3D) -> str:
    atomnames = m.atom_names
    atomneighbors = m.atom_neighbors.numpy()
    bondlengths = m.bond_lengths.numpy()
    bondpairs = m.bond_pairs.numpy()
    if m.num_atoms > 2:
        angles = m.angles.numpy()
        angletrios = m.angle_trios.numpy()
        if m.num_atoms > 3:
            dihedrals = m.dihedrals.numpy()
            dihedralquartets = m.dihedral_quartets.numpy()
    new_z_matrix = ""
    found_dihedrals = []

    # First atom

    new_z_matrix += atomnames[0]
    new_z_matrix += "\n"

    # From second atom

    for i in range(1, m.num_atoms):

        nb_i = atomneighbors[i][0]
        bd_idx_i = tf.where(
            tf.reduce_all(
                tf.equal(
                    bondpairs,
                    tf.constant(
                        sorted([i, nb_i]),
                        dtype=tf.int64
                    )
                ),
                axis=1
            )
        ).numpy()[0][0]

        bd_i = bondlengths[bd_idx_i]
        bd = '{:>11.5f}'.format(bd_i)

        if i == 2:
            anb_i = atomneighbors[nb_i][0]

            a_idx_i = tf.where(
                tf.reduce_all(
                    tf.equal(
                        angletrios,
                        tf.constant(
                            sorted([i, nb_i, anb_i]),
                            dtype=tf.int64
                        )
                    ),
                    axis=1
                )
            ).numpy()[0][0]

            a_i = angles[a_idx_i]
            ad = '{:>11.5f}'.format(a_i)

        if i > 2:
            found = False
            found_quartet = None
            for l in range(len(atomneighbors[i])):
                nb_i = atomneighbors[i][l]
                if nb_i > i:
                    continue
                if found:
                    break

                for j in range(len(atomneighbors[nb_i])):
                    anb_i = atomneighbors[nb_i][j]
                    if anb_i == i or anb_i > i:
                        continue
                    if found:
                        break

                    previous = [i, nb_i, anb_i]
                    for k in range(len(atomneighbors[anb_i])):
                        dnb_i = atomneighbors[anb_i][k]
                        sorted_quartet = tuple(sorted([i, nb_i, anb_i, dnb_i]))
                        if dnb_i not in previous and sorted_quartet not in found_dihedrals and dnb_i < i:
                            found_quartet = {"i": i, "nb_i": nb_i, "anb_i": anb_i, "dnb_i": dnb_i}
                            found = True
                            break

            if not found:
                matching_dihedrals = np.where(dihedralquartets[:, -1] == i)[0]
                if len(matching_dihedrals) == 0:
                    raise IndexError("Could not detect dihedrals!")
                else:
                    first_matching_dihedral = dihedralquartets[matching_dihedrals[0]]
                    found_nb = False
                    for atom in first_matching_dihedral:
                        if atom in atomneighbors[i]:
                            nb_i = atom
                            found_nb = True
                            break
                    if not found_nb:
                        raise IndexError("Could not detect dihedrals!")
                    else:
                        found_anb = False
                        for atom in first_matching_dihedral:
                            if atom == i:
                                continue
                            elif atom in atomneighbors[nb_i]:
                                anb_i = atom
                                found_anb = True
                                break
                        if not found_anb:
                            raise IndexError("Could not detect dihedrals!")
                        else:
                            for atom in first_matching_dihedral:
                                if atom == i or atom == nb_i or atom == anb_i:
                                    continue
                                else:
                                    dnb_i = atom
                                    break
                found_quartet = {"i": i, "nb_i": nb_i, "anb_i": anb_i, "dnb_i": dnb_i}
                sorted_quartet = sorted(
                    np.array([i, found_quartet["nb_i"], found_quartet["anb_i"], found_quartet["dnb_i"]]))
                sorted_trio = sorted(np.array([i, found_quartet["nb_i"], found_quartet["anb_i"]]))
                sorted_duo = sorted(np.array([i, found_quartet["nb_i"]]))

            else:
                sorted_quartet = sorted(
                    np.array([i, found_quartet["nb_i"], found_quartet["anb_i"], found_quartet["dnb_i"]]))
                sorted_trio = sorted(np.array([i, found_quartet["nb_i"], found_quartet["anb_i"]]))
                sorted_duo = sorted(np.array([i, found_quartet["nb_i"]]))

            nb_i = found_quartet["nb_i"]
            anb_i = found_quartet["anb_i"]
            dnb_i = found_quartet["dnb_i"]

            bd_idx_i = tf.where(
                tf.reduce_all(
                    tf.equal(
                        bondpairs,
                        tf.constant(
                            sorted_duo,
                            dtype=tf.int64
                        )
                    ),
                    axis=1
                )
            ).numpy()[0][0]

            bd_i = bondlengths[bd_idx_i]
            bd = '{:>11.5f}'.format(bd_i)

            a_idx_i = tf.where(
                tf.reduce_all(
                    tf.equal(
                        angletrios,
                        tf.constant(
                            sorted_trio,
                            dtype=tf.int64
                        )
                    ),
                    axis=1
                )
            ).numpy()[0][0]

            a_i = angles[a_idx_i]
            ad = '{:>11.5f}'.format(a_i)

            d_idx_i = tf.where(
                tf.reduce_all(
                    tf.equal(
                        dihedralquartets,
                        tf.constant(
                            sorted_quartet,
                            dtype=tf.int64
                        )
                    ),
                    axis=1
                )
            ).numpy()[0][0]

            d_i = dihedrals[d_idx_i]
            found_dihedrals.append(tuple(sorted_quartet))

            dd = '{:>11.5f}'.format(d_i)
            dihedral_string = '{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(atomnames[i], nb_i + 1, bd,
                                                                                          anb_i + 1, ad, dnb_i + 1, dd)
            new_z_matrix += dihedral_string
            new_z_matrix += "\n"

        elif i == 1:

            bond_string = '{:<3s} {:>4d}  {:11s}'.format(atomnames[i], nb_i + 1, bd)
            new_z_matrix += bond_string
            new_z_matrix += "\n"

        elif i == 2:

            angle_string = '{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(atomnames[i], nb_i + 1, bd, anb_i + 1, ad)
            new_z_matrix += angle_string
            new_z_matrix += "\n"

    positions, atoms = _read_zmatrix(new_z_matrix)

    new_xyz = positions
    for i in range(len(atomnames)):
        atom_idx = int(i)
        cds = new_xyz[i]
        point = Point3D(float(cds[0]), float(cds[1]), float(cds[2]))
        m.conformer.SetAtomPosition(atom_idx, point)

    return new_z_matrix


def convert_molfile_to_zmatrix(molfile: str) -> str:
    if molfile.endswith(".mol"):
        molblock = Chem.MolToMolBlock(Chem.MolFromMolFile(molfile))
    else:
        molblock = molfile

    m = canonicalize_3d_molecule(MolecularGraph3D(molblock, 2, True))
    zm = create_z_matrix(m)

    return zm


def create_3d_graph_from_smiles(smi: str) -> MolecularGraph3D:
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol)
    m = canonicalize_3d_molecule(MolecularGraph3D(Chem.MolToMolBlock(mol), 2))

    return m


def convert_smiles_to_zmatrix(smi: str) -> str:
    m = create_3d_graph_from_smiles(smi)
    zm = create_z_matrix(m)

    return zm


def create_3d_mol_from_smiles(smi: str) -> Mol:
    m = create_3d_graph_from_smiles(smi)

    return m.mol
