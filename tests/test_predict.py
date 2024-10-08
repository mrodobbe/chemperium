import chemperium as cp


def test_thermo_2d():
    pass
    # smi = "COc1ccccc1"
    # thermo = cp.training.predict.Thermo("g3mp2b3", "2d", "test_data")
    # thermo.predict_entropy(smi)


def test_thermo_3d():
    pass
    # smi = "COc1ccccc1"
    # xyz = '16\n' \
    #       'name=C7H8O_367 charge=0 multiplicity=1 smiles=COc1ccccc1 crc=-1669956209 parent_crc=365106234\n' \
    #       'C          2.76930        0.32250       -0.00050\n' \
    #       'O          1.76340       -0.67620       -0.00000\n' \
    #       'C          0.45600       -0.27750       -0.00000\n' \
    #       'C         -0.49220       -1.31180       -0.00020\n' \
    #       'C         -1.84930       -1.01160       -0.00010\n' \
    #       'C         -2.28360        0.31900        0.00010\n' \
    #       'C         -1.33830        1.34160        0.00020\n' \
    #       'C          0.03130        1.05620        0.00020\n' \
    #       'H          3.72200       -0.21080       -0.00090\n' \
    #       'H          2.71000        0.95720       -0.89500\n' \
    #       'H          2.71080        0.95730        0.89390\n' \
    #       'H         -0.13750       -2.33800       -0.00030\n' \
    #       'H         -2.57430       -1.82150       -0.00030\n' \
    #       'H         -3.34470        0.55070        0.00010\n' \
    #       'H         -1.65940        2.38020        0.00040\n' \
    #       'H          0.74700        1.87060        0.00030'
    # llot = -5.13245
    # thermo = cp.training.predict.Thermo("g3mp2b3", "3d", "test_data")
    # thermo.predict_enthalpy(smi, xyz, llot, t=1000, quality_check=True)


def test_liquid_2d():
    pass
    # smi = "COc1ccccc1"
    # liquid = cp.training.predict.Liquid("logp", "2d", "test_data")
    # liquid.predict(smi)
