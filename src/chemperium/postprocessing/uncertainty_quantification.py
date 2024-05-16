import json
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import Chem
from typing import List


def add_reliability_score(smiles: List[str],
                          dataset: str,
                          uncertainty: List[float],
                          molecules: List[Mol]) -> List[str]:
    with open(f"{dataset}/fact_sheet.json") as json_file:
        fact_sheet = json.load(json_file)

    scores = []

    for i in range(len(smiles)):
        smi = smiles[i]
        unc = uncertainty[i]
        mol = molecules[i]

        bm = get_scaffold(Chem.MolFromSmiles(smi))
        nh = mol.GetNumHeavyAtoms()
        atom_type = True
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym == "H":
                continue
            if sym not in fact_sheet["atom_type"]:
                atom_type = False
                break

        score = 0
        if bm in fact_sheet["bm_scaffold"]:
            score += 1
        if nh <= fact_sheet["num_heavy"]:
            score += 1
        if atom_type:
            score += 1
        if unc < fact_sheet["uncertainty"]:
            score += 1

        scores.append(f"{score}*")

    return scores


def get_scaffold(mol: Mol) -> str:
    """
    Get the Murcko scaffold of a molecule
    :param mol: RDKit Mol object
    :return: SMILES string
    """
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    scaffold = GetScaffoldForMol(mol)
    smi = Chem.MolToSmiles(scaffold)

    return smi
