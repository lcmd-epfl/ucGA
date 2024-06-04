import numpy as np
from navicatGA.wrappers_smiles import smiles2mol_structure 
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir,'SA_Score'))
import sascorer
import ucGA.utils.scscore
import ucGA.utils.scscore.scscore.standalone_model_numpy as standalone_model_numpy

def synthetic_difficulty_score(smiles,config,scscore_path,n=4,factor=6):
    """
    Calculate the synthetic difficulty score for a given SMILES string.

    The synthetic difficulty score is a contribution of three components:
    1. SC-Score (synthetic complexity score),
    2. SA-Score (synthetic accessibility score), and
    3. The number of heavy atoms in the molecule.

    Each component is normalized using a threshold function before averaging.

    Parameters:
    - smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string representing a molecule.
    - config (Config): An instance of the 'Config' class containing configuration settings like cutoff thresholds for scoring functions.

    
    Returns:
    - float: The average synthetic difficulty score for the given SMILES string. A lower score indicates easier synthesis.
    """
    
    # Load SCScore model
    model_scs = standalone_model_numpy.SCScorer()
    model_scs.restore(os.path.join(scscore_path, 'models', 'full_reaxys_model_1024uint8', 'model.ckpt-10654.as_numpy.json.gz'))
    
    #Calculate SCScore,SAScore and Nr. of heavy atoms
    (smi_scs, sco_scs) = model_scs.get_score_from_smi(smiles)
    mol_rdkit = smiles2mol_structure(smiles)
    sco_sa = sascorer.calculateScore(mol_rdkit)
    nr_heavy_atoms = mol_rdkit.GetNumHeavyAtoms()

    # Apply the threshold function
    sco_sa_norm = cutoff_func_small_minimize(sco_sa,config.sascore_cutoff,n=2,factor=0.5)           
    size_score_norm = cutoff_func_small_minimize(nr_heavy_atoms,config.heavy_atom_count_cutoff)
    sco_scs_norm = cutoff_func_small_minimize(sco_scs,config.scscore_cutoff,n=2,factor=0.2)
 
    # Average the three contributions

    synth_score_average = float(np.average([sco_scs_norm,sco_sa_norm,size_score_norm]))
    
    return synth_score_average
    
    
def cutoff_func_small_minimize(x,cutoff,n=4,factor=6):
    return (1-cutoff_func(x,cutoff,n,factor))


def cutoff_func(x,cutoff,n=4,factor=6):
    """
    Applies a custom cutoff function to a given value.

    This function computes a scaling factor based on the distance of the input value 
    from a specified cutoff threshold. It returns a scaled value between 0 and 1, 
    where values less than the cutoff are scaled to 1, and values greater than the 
    cutoff are scaled down based on the specified parameters.

    Parameters:
    x (float): The input value to be scaled.
    cutoff (float): The cutoff threshold.
    n (int, optional): The power to which the difference is raised. Defaults to 4.
    factor (int, optional): The divisor for scaling the difference. Defaults to 6.

    Returns:
    float: The scaled value, between 0 and 1.
    """
    if x <= cutoff: return 1
    else: return 1 / (np.sqrt(1+ ((x-cutoff)/factor )**n)) 
