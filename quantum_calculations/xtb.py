import hashlib
import os
import numpy as np
import ray
from pathlib import Path
from navicatGA.wrappers_smiles import (
    smiles2mol_structure)
from pysisyphus.calculators.XTB import XTB as pysis_XTB
from pysisyphus.helpers import geom_from_xyz_file


from ucGA.utils.wrapper import Wrapper
from ucGA.quantum_calculations.slatm import generate_single_slatm


class XTB_Processor:
    def __init__(self, smiles, config, output_suffix="Structures"):
        self.config = config
        self.smiles = smiles
        
        self.structure_path = os.path.join(self.config.output_path, output_suffix)
        self.smiles_hash = self.generate_smiles_hash()

        self.mol2xyz_path = f"{self.structure_path}/{self.smiles_hash}.xyz"
        self.pysis_conv_path = f"{self.structure_path}/{self.smiles_hash}_pysis_conv.xyz"

        self.wrapped_mol2xyz = Wrapper("mol2xyz", "navicatGA.wrappers_smiles")
        self.smiles = smiles

    def create_xyz_with_xtb(self):
        """
        Creates xyz file using xTB

        Returns:
        None
        """
        
     
        mol_rdkit = smiles2mol_structure(self.smiles)
        self._convert_mol_to_xyz(mol_rdkit, self.mol2xyz_path)
        

        pysis_opt = self._perform_xTB_optimization(self.mol2xyz_path)
        if pysis_opt is None:
            raise ValueError("Optimization failed.")

        self._write_pysis_converted_coords(pysis_opt)

    def generate_single_slatm(self):
            generate_single_slatm(self.pysis_conv_path, self.config.mbtypes_path, self.config.sizeof_slatm)
        

    def generate_smiles_hash(self):
        return hashlib.md5(bytes(self.smiles, encoding='utf-8')).hexdigest()

    def _convert_mol_to_xyz(self, mol_rdkit, filepath):
        self.wrapped_mol2xyz(mol_rdkit, filepath)

    def _perform_xTB_optimization(self, filepath):
        mol_pysis = geom_from_xyz_file(filepath)
        pysis_calc = pysis_XTB(gfn=2, acc=1.0)        
        return pysis_calc.run_opt(mol_pysis.atoms, mol_pysis.coords)

    def _write_pysis_converted_coords(self, pysis_opt):
        pysis_prepared_coords = pysis_XTB.prepare_coords(pysis_opt, pysis_opt.opt_geom.atoms, pysis_opt.opt_geom.coords)
        with open(self.pysis_conv_path, "w") as f:
            f.write(pysis_prepared_coords)

# Usage example
#config = Config()  # Ensure Config is properly defined
#molecule_processor = MoleculeProcessor(config)
#molecule_processor.create_xyz_with_xtb(smiles_string, temp_subpath)

    