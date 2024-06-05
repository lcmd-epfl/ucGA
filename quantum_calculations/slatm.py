import os
from pathlib import Path
import numpy as np
import psutil
import qml
from qml.representations import get_slatm_mbtypes

def generate_single_slatm(path_xyz, mbtypes_path, sizeof_slatm):
    compounds = [qml.Compound(path_xyz)]
    path_folder = Path(path_xyz).parent
    filename = Path(path_xyz).stem

    if os.path.exists(mbtypes_path):
        mbtypes = np.load(mbtypes_path, allow_pickle=True)
    else:
        mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
        np.save(mbtypes_path, mbtypes)

    X = np.zeros((len(compounds), sizeof_slatm), dtype=np.float16)
    names = []

    for i, mol in enumerate(compounds):
        mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
        X[i, :] = np.float16(mol.representation)
        names.append(mol.name)

    np.save(os.path.join(path_folder, f"repr{filename}.npy"), X)
    np.save(os.path.join(path_folder, f"names{filename}.npy"), names)
    
    
