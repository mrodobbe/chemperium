import os
import sys
from operator import itemgetter
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry.rdGeometry import Point3D
from chemperium.features.calc_features import periodic_table
import numpy as np
sys.path.insert(0, os.path.split(__file__)[0])

p5 = Chem.MolFromSmarts('[H][O,#7,Sv2]~*~*~[#8,#7v3]')
p6 = Chem.MolFromSmarts('[H][O,#7,Sv2]~*~*~*~[#8,#7v3]')
p1 = Chem.MolFromSmarts('[#1;$(*[#8,#7,#16])]')
inv_pt = {v: k for k, v in periodic_table().items()}


def find_h_bonds(mol, xyz_lines, smi):
    """identifies intra and intermolecular H-bonds in 3D.
    It calculates Gasteiger charges and if those are >0.15 and H or <-0.15 and (O,N,S,F)
    then we have the donor and acceptor atoms. If there is no H-bond donor
    returns [0,0] otherwise [n_intra,n_inter] to be used for classification.
    Intramolecular H-bonds are identified by measuring the distance between the
    pier atoms and if this <3.5 A, then it is intramolecular. Otherwise, it is intermolecular."""
    ret = {
        'H_intra': [],
        'H_inter': [],
        'acc_intra': [],
        'acc_inter': [],
        'don_intra': [],
        'don_inter': []
    }
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ret

    lines = xyz_lines.split("\n")[2:]
    newies = [line.split(" ") for line in lines]
    cleans = []
    for i in newies:
        g = [x for x in i if x]
        cleans.append(g)
    coords = [x[1:] for x in cleans]
    coords = np.asarray(coords).astype(np.float64)
    ats = [x[0] for x in cleans]
    m1 = Chem.AddHs(m)
    AllChem.EmbedMolecule(m1)
    try:
        c1 = m1.GetConformer()
    except ValueError:
        m1 = mol
        c1 = mol.GetConformer()
    for i in range(m1.GetNumAtoms()):
        x, y, z = coords[i]
        c1.SetAtomPosition(i, Point3D(x, y, z))
        atom = m1.GetAtomWithIdx(i)
        an = inv_pt.get(ats[i])
        atom.SetAtomicNum(an)

    m = m1

    rdPartialCharges.ComputeGasteigerCharges(m)
    don = []
    acc = []
    hbonds = [0, 0]
    atoms = m.GetAtoms()
    charges = []
    for i, at in enumerate(atoms):
        #         chg = eval(at.GetProp('_GasteigerCharge'))
        try:
            chg = eval(at.GetProp('_GasteigerCharge'))
        except:
            chg = 0.0
        # end try
        charges.append(chg)
    # end for
    for i, at in enumerate(atoms):
        chg = charges[i]
        if at.GetSymbol() == 'H' and chg > 0.1:
            don.append([at.GetNeighbors()[0].GetIdx(), at.GetIdx()])
            # gets the id of the atom to which this H is bound to
        elif at.GetSymbol() in ['O', 'N', 'S', 'F'] and chg < -0.15:
            acc.append(i)
        # end if
    # end for
    if len(don) == 0:
        ret['acc_inter'] = acc
        return ret
    # end if
    conf = m.GetConformer(0)
    hb_intra = []
    hb_inter = []
    acc_intra = []
    don_intra = []
    for did, hid in don:
        # ii=0
        dd = []
        for aid in acc:
            if did != aid:  # then measure distance
                p1 = conf.GetAtomPosition(did)
                p2 = conf.GetAtomPosition(aid)
                p3 = conf.GetAtomPosition(hid)
                dist = p2.Distance(p3)  # distance between H on the donor to the acceptor
                angle = rdMolTransforms.GetAngleDeg(conf, did, hid, aid)  # angle for donor-H-acceptor
                ids = [r[0] for r in don]
                # list of donor ids  (don is a list of lists, an element is [donor_id, hydrogen_id])
                if dist < 2.5 and angle > 100:
                    if aid in ids:
                        # if this acceptor is donor too
                        # then measure the minimum distance between its hydrogens and the current donor's hydrogen,
                        # take the minimum
                        dst = []
                        p1 = conf.GetAtomPosition(hid)  # position of donor's hydrogen
                        for aa in atoms[aid].GetNeighbors():
                            if aa.GetSymbol() == 'H':
                                p2 = conf.GetAtomPosition(aa.GetIdx())
                                dst.append(p1.Distance(p2))
                            # end if
                        # end for
                        dst = min(dst)  # distance between acceptor's hydrogen to donor's hydrogen
                    else:
                        dst = 1.05 * dist
                    # end if
                    if dst > dist:
                        dd.append([aid, dist])
                        acc_intra.append(aid)
                        don_intra.append(did)
                    # end if
                # end if
            # end if
        # end for
        if len(dd) == 0:
            hbonds[1] += 1
            hb_inter.append(hid)
        else:
            if len(dd) > 1:
                dd = sorted(dd, key=itemgetter(1))
            # end if
            hbonds[0] += 1
            hb_intra.append(hid)
        # end if
    # end for
    ret = dict()
    ret['H_intra'] = hb_intra
    ret['H_inter'] = hb_inter
    ret['acc_intra'] = acc_intra
    ret['acc_inter'] = list(set(acc) - set(acc_intra))
    ret['don_intra'] = don_intra
    ret['don_inter'] = list(set([d[0] for d in don]) - set(don_intra))
    return ret
