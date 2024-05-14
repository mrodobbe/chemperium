import numpy as np
from scipy.optimize import curve_fit
from rdkit.Chem.rdchem import Mol


def cp_fit(temperatures, a1, a2, a3, a4, a5):
    return a1 + a2 * temperatures + a3 * temperatures ** 2 + a4 * temperatures ** 3 + a5 * temperatures ** 4


def enthalpy_fit(temperatures, a1, a2, a3, a4, a5, a6):
    return a1 * temperatures + (a2 / 2) * temperatures ** 2 + (a3 / 3) * temperatures ** 3 + \
        (a4 / 4) * temperatures ** 4 + (a5 / 5) * temperatures ** 5 + a6


def entropy_fit(temperatures, a1, a2, a3, a4, a5, a7):
    return a1 * np.log(temperatures) + a2 * temperatures + (a3 / 2) * temperatures ** 2 + \
        (a4 / 3) * temperatures ** 4 + (a5 / 5) * temperatures ** 5 + a7


def get_cp_coefficients(temperatures, cp_values):
    if len(cp_values.shape) > 1:
        coefficients = []
        for i in range(cp_values.shape[0]):
            popt, _ = curve_fit(cp_fit, temperatures, cp_values[i])
            coefficients.append(popt)
        coefficients = np.array(coefficients).T
    else:
        popt, _ = curve_fit(cp_fit, temperatures, cp_values)
        coefficients = np.array(popt)
    return coefficients


def get_nasa_coefficients(temperatures, h298, s298, cp_values):
    a1, a2, a3, a4, a5 = get_cp_coefficients(temperatures, cp_values * (4.184 / 8.314))
    a6 = h298 * (4.184 / 8.314) - enthalpy_fit(298.15, a1, a2, a3, a4, a5, 0)
    a7 = s298 * (4.184 / 8.314) - entropy_fit(298.15, a1, a2, a3, a4, a5, 0)
    if len(cp_values.shape) > 1:
        return np.array([a1, a2, a3, a4, a5, a6, a7]).T
    else:
        return np.array([a1, a2, a3, a4, a5, a6[0], a7[0]])


def get_chemkin_file(name: str, smiles: str, method: str, mol: Mol, nasa_coefficients: np.ndarray):
    atom_dict = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in atom_dict:
            atom_dict[sym] += 1
        else:
            atom_dict[sym] = 1
    if len(atom_dict) > 4:
        raise ValueError("Chemkin format is only valid for molecules with up to 4 atom types.")
    else:
        num_types = len(atom_dict)

    chemkin = f"!\n! Filename: {name}\n! Smiles: {smiles}\n! method: {method}\n"
    chemkin += f"{name.upper(): <18}"
    chemkin += "      "
    for atom_type in atom_dict:
        chemkin += f"{atom_type: <2}{atom_dict[atom_type]: <3}"
    for i in range(4 - num_types):
        chemkin += "     "
    chemkin += "G   200.00    5000.00   1500.00    1\n"
    for i in range(5):
        formatted_coefficient = format_array(np.array(nasa_coefficients[i]))
        chemkin += formatted_coefficient
    chemkin += "    2\n"
    formatted_coefficient = format_array(np.array(nasa_coefficients[5]))
    chemkin += formatted_coefficient
    formatted_coefficient = format_array(np.array(nasa_coefficients[6]))
    chemkin += formatted_coefficient
    for i in range(3):
        formatted_coefficient = format_array(np.array(nasa_coefficients[i]))
        chemkin += formatted_coefficient
    chemkin += "    3\n"
    for i in range(3, len(nasa_coefficients)):
        formatted_coefficient = format_array(np.array(nasa_coefficients[i]))
        chemkin += formatted_coefficient
    chemkin += "                   4\n"
    return chemkin


def format_array(arr, decimals=8):
    formatted_arr = np.array2string(arr, formatter={'float_kind': lambda x: ("{: ." + str(decimals) + "e}").format(x)})
    return formatted_arr.replace('\n', ', ')
