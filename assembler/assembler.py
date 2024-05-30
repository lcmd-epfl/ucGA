from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import Draw
from rdkit.Chem import Recap,BRICS
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import ReplaceCore, GetMolFrags, ReplaceSubstructs, CombineMols
import numpy as np
import rdkit
import pandas as pd

def get_connect_idx(mol):
    list_connect =([x.GetAtomicNum() for x in mol.GetAtoms()])
    return [i for i, e in enumerate(list_connect) if e == 0]

def combine_fragments(chromosome,connections):
    """
    Combine molecular fragments specified by the chromosome using connection information.

    Parameters:
    - chromosome (list): List of SMILES strings representing molecular fragments.
    - connections (list): List specifying the connections between fragments, e.g. [[1,2],[0],[0]] for three fragments

    Returns:
    str: Molecular SMILES string obtained by combining the specified fragments.

    The function takes a list of SMILES strings (chromosome) representing molecular fragments and
    combines them according to the connection information provided in the connections list. It builds
    the combined molecule step by step, handling wildcard atoms and creating bonds between fragments.
    The final assembled molecular SMILES string is returned.

    Note: The function uses RDKit for molecular manipulations.

    """
    # Initialize variables
    nr_genes=len(chromosome)
    mol_fragments= []

    # Convert SMILES strings to list of RDKit Mol objects
    for gene_index in range(nr_genes):
        mol_fragments.append(Chem.MolFromSmiles(chromosome[gene_index]))
    
    # Get connection indices for each fragment
    connect_idx = [get_connect_idx(mol_fragments[i]) for i in range(len(mol_fragments))]
    print(connect_idx)
    
     # Initialize variables for molecule assembly   
    mol_size_correct =0
    fragment_index=0
    comb_mol=mol_fragments[0]

    # Get fragment sizes for cumulative sum calculations
    fragment_sizes = [mol_fragments[i].GetNumAtoms() for i in range(len(mol_fragments)) ]
    sum_size = list(np.cumsum(fragment_sizes)) #-fragment_sizes[0]
    sum_size.insert(0, 0)
    
    # Main loop for assembling fragments    
    for fragment_index in range(1,len(mol_fragments)):
        print("fragment index", fragment_index)
        previous_size = comb_mol.GetNumAtoms()
        comb_mol = CombineMols(comb_mol,mol_fragments[fragment_index])
        comb_mol = rdkit.Chem.RWMol(comb_mol)

        # Identify the atom index of the bond site in the new fragment
        fragment_atom_pos=1000
        for i in range(len(connections[fragment_index])):
            if connections[fragment_index][i] < fragment_index:
                fragment_atom_pos=i #check!!!

        # Calculate atom index in the new fragment
        fragment_atom_idx = connect_idx[fragment_index][fragment_atom_pos] + previous_size 
        
        # Identify the neighbor of the wildcard
        fragment_atom_neighbor_idx = comb_mol.GetAtoms()[fragment_atom_idx].GetNeighbors()[0].GetIdx()

        # Identify the index of the base fragment
        base_molecule = connections[fragment_index][fragment_atom_pos]
        base_atom_pos = 1000
        for i in range(len(connections[base_molecule])):
            if connections[base_molecule][i] == fragment_index:
                base_atom_pos=i
                print('found base_atom_pos',base_atom_pos)

        # Calculate the atom index in the base fragment

        base_atom_idx = connect_idx[base_molecule][base_atom_pos] + sum_size[base_molecule] #is index of base atom in mol_comb


        # Identify the neighbor of the wildcard in the base fragment
        list_temp=comb_mol.GetAtoms()
        base_atom_neighbor_idx = comb_mol.GetAtoms()[int(base_atom_idx)].GetNeighbors()[0].GetIdx()

        comb_mol.AddBond(int(fragment_atom_neighbor_idx),base_atom_neighbor_idx,order=Chem.rdchem.BondType.SINGLE)

    
    #cleaning up the wildcards
    index =0
    while index < len(comb_mol.GetAtoms()):
        #print(index)
        atom = comb_mol.GetAtoms()[int(index)]
        atom_type = atom.GetSymbol()
        #print(atom_type)
        if atom_type=='*':
            comb_mol.RemoveAtom(atom.GetIdx())
            #print("removed")
            
        else:
            index +=1
            
    # Return the assembled SMILES string
    return Chem.MolToSmiles(comb_mol)

