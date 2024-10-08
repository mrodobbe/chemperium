from chemperium.gaussian.zmatrix import *


def test_zmatrix():
    zm = convert_smiles_to_zmatrix("CCCC")
    print(zm)
