import numpy as np

import networkx as nx
from rdkit import Chem
import torch

def _normalize_img(img, p = 0.0028, min_pixel = 0):

    p = p / 100

    n_pixels = img.shape[0] * img.shape[1] * p
    n_pixels = int(np.around(n_pixels))

    idxs = np.argpartition(img, -n_pixels, axis = None)[-n_pixels: ]
    idxs = np.array(np.unravel_index(idxs, img.shape)).T

    values = [img[i, j] for i, j in idxs]
    thresh = min(values)

    img = ((img.astype(float) - min_pixel) * (img < thresh))
    scaled_img = img * (256. / (thresh - min_pixel))
    scaled_img[scaled_img > 255] = 255

    scaled_img = scaled_img.astype(np.uint8)
    # for i, j in idxs: img[i, j] = 255

    return scaled_img

def normalize_img(img, p = 0.0028, min_pixel = 0):
    if len(img.shape) == 2: 
        return _normalize_img(img, p, min_pixel)
    elif len(img.shape) in [3, 4]:
        flattened = img.reshape(-1, img.shape[-2], img.shape[-1])
        normalized = np.array([_normalize_img(img_) for img_ in flattened])
        normalized = normalized.reshape(*img.shape)
        return normalized
    else:
        raise ValueError("The dimensions of input must be either 2, 3, or 4!")

def onehot(k,  n):
    return np.eye(n)[k]

def mol_feature(**kwargs):

    atoms_map = {
        5: 0, # B
        6: 1, # C
        7: 2, # N
        8: 3, # F
        14: 4, # Si
        15: 5, # P
        16: 6, # S
        17: 7, # Cl
        33: 8, # As
        34: 9, # Se
        35: 10, # Br
        52: 11, # Te
        53: 12, # I
        85: 13 # At
    }

    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4
    } # sp, sp2 , sp3 , sp3 d, sp3 d2 ,

    chirality_map = {
        "CHI_TETRAHEDRAL_CW": 0,
        "CHI_TETRAHEDRAL_CCW ": 0,
        "CHI_TETRAHEDRAL": 0
    } 

    # Atoms
    try: atom = onehot(atoms_map[kwargs['atom']], 15)
    except KeyError: atom = onehot(14, 15)

    # Covalent Bonds
    covalent = onehot(kwargs['covalent'], 6)

    # Charge
    charge = np.array([kwargs['charge']])

    # radical electrons
    electrons = np.array([kwargs['electrons']])

    # hybridization
    try: hybrid = onehot(hybridization_map[kwargs['hybrid']], 6)
    except KeyError: hybrid = onehot(5, 6)

    # aromaticity
    aromatic = onehot(int(kwargs['aromatic']), 2)

    # Hydrogen
    hydrogen_explicit = onehot(kwargs['hydrogen_explicit'], 6)
    hydrogen_implicit = onehot(kwargs['hydrogen_implicit'], 6)

    # chirality type
    try: chi_type = onehot(chirality_map[kwargs['chi_type']], 2)
    except KeyError: chi_type = onehot(1, 2)

    feature = np.concatenate([atom, covalent, charge, electrons, hybrid, aromatic, hydrogen_explicit, hydrogen_implicit, chi_type], axis = 0)
    feature = torch.from_numpy(feature).to(torch.float32)
    return feature


def process_molecule(smile):

    '''
    The procedure to process molecules as described by xiong et al., 2019

    atoms: B, C, N, O, F, Si, P, S, Cl, As, Se, Br, Te, I, At, metal
    number of covalent bonds: 0,1,2,3,4,5
    electrical charge: integer
    number of radical electrons: integer
    hybridization: sp, sp2, sp3, sp3d, sp3d2, other
    aromaticity: 0, 1
    number of connected hydrogens: 0, 1, 2, 3, 4, 5
    chirality type: tetrahedron, N/A
    
    Total vector size: 45

    Args: 
    SMILES string, str

    Returns:
    a tuple containing a networkx object and a feature vector

    '''

    G = nx.Graph()
    atom_features = []
    mol = Chem.MolFromSmiles(smile)
    
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i + 1)
        descriptor = {
            "atom": atom.GetAtomicNum(),
            "covalent": atom.GetTotalDegree(),
            "charge": atom.GetFormalCharge(),
            "electrons": atom.GetNumRadicalElectrons(),
            "hybrid": atom.GetHybridization(),
            "aromatic": atom.GetIsAromatic(),
            "hydrogen_implicit": atom.GetNumImplicitHs(),
            "hydrogen_explicit": atom.GetNumExplicitHs(),
            "chi_type": atom.GetChiralTag()
        }

        atom_features.append(mol_feature(**descriptor))
    
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1)

    return (G, atom_features)
    
