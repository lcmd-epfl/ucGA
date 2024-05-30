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
    
    
    
def generate_SLATM_from_list(path,SIZEOFSLATM,namelist_xyz,appendix=''):
    namelist=[]
    for xyz in namelist_xyz:
        namelist.append(qml.Compound(path+xyz))
    compounds = np.asarray(namelist, dtype=object)  # WARNING: REMOVE SLICING
    print("compunds",compounds,flush=True)

    print("Generated compounds; RAM memory % used:", psutil.virtual_memory()[2], flush=True)
    print("Total RAM:", psutil.virtual_memory()[0], flush=True)
    print("Available RAM:", psutil.virtual_memory()[1], flush=True)
    mb_path="/home/student7/LucaSchaufelberger/MasterThesis/FORMED_ML/predict/data_test/mbtypes.npy"
    if os.path.exists(mb_path):
        mbtypes = np.load(mb_path, allow_pickle=True)
        print('found mbtypes')
    else:
        mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in compounds])
        mbtypes = np.array(mbtypes)
        np.save("mbtypes.npy", mbtypes)
        print('mbtypes not found')
    # replace this number with the size of the mbtypes
    
    X = np.zeros((len(compounds), SIZEOFSLATM), dtype=np.float16)
    N = []
    print(
        "Generated empty representation matrix; RAM memory % used:",
        psutil.virtual_memory()[2],
        flush=True,
    )
    for i, mol in enumerate(compounds):
        print(f"Tackling representation of {namelist[i]}", flush=True)
        print("mol",mol)
        mol.generate_slatm(mbtypes, local=False, dgrids=[0.1, 0.1])
        print(mol.representation.shape)
        X[i, :] = np.float16(mol.representation)
        N.append(mol.name)
        print(
            "Filled in one representation vector; RAM memory % used:",
            psutil.virtual_memory()[2],
            flush=True,
        )
        del mol

    N = np.array(N)
    np.save(path+"/repr"+appendix+".npy", X)
    np.save(path+"/names"+appendix+".npy", N)