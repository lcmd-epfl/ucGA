import os
import pickle
from pathlib import Path
import pandas as pd

class Config:
    def __init__(self, output_path,lambda_cb):
        self.config_dir = Path(__file__).parent        

        
        self.lambda_cb = lambda_cb  # confidence bound (explorative), use -2 for exploitative, +2 for explorative
        
        # Basic genetic algorithm parameters
        self.n_genes = 3  # Nr. of genes
        self.pop_size = 48 #48  # Population Size
        self.max_gen = 50  # 



        # Iterations and UCB settings
        self.nr_ml_runs_per_iteration = 50  
        
        self.nr_ucb_recalculate = 8  # In the end, perform DFT on these
        self.bucket_size_chemberta = 16  # Bucket size for batch evaluation in the ChemBERTa SMILES model
        self.nr_evals_per_chrom = 6  # Nr. of evaluations of the extended ensemble
        self.nr_evals_SMILES = 4  # Nr. of SMILES model evaluation in the extended ensemble
        self.nr_evals_SLATM = self.nr_evals_per_chrom - self.nr_evals_SMILES

        # Directory and file handling
        self.output_path = Path(output_path)
  
        self.structures_path = self.output_path / 'Structures' #TODO # Folder where forcefield and xTB calculations are saved
        os.makedirs(self.structures_path, exist_ok=True)
        os.makedirs(self.output_path / 'models', exist_ok=True)        
        os.makedirs(self.output_path / 'UCB', exist_ok=True)
        with open(self.output_path / 'Fitness_logger.csv', 'w') as file:
            file.write('')
        
        
        # Load cores and subst data from csv files
        df_cores = pd.read_csv(self.config_dir.parent /  "Fragment_Pool/cores.csv")
        self.cores = df_cores["Smiles"].values.tolist()
        
        df_subst = pd.read_csv(self.config_dir.parent /  "Fragment_Pool/substituents.csv")
        self.subst = df_subst["Smiles"].values.tolist()

        # Set up gene pool (core and substituent lists)
        self.alphabet_list = [list(self.cores), list(self.subst), list(self.subst)]
        self.connections = [[1, 2], [0], [0]] #Connectivity of the fragments, e.g. fragment 0 (core) is connected to fragments [1,2] (substituents)
        """
        # For bug-fixing, use smaller gene pool
        self.alphabet_list = [["*O*","*C*"],["O*","C*"],["O*","C*"]]
        self.connections = [[1, 2], [0], [0]]
        """
      
        
        # Parameters for Chimera         
        self.tolerances = [0.2, 0.3, 0.3]
        self.absolutes = [False, False, False]

        # Parameters for Synthetic Difficulty score
        self.heavy_atom_count_cutoff = 35
        self.scscore_cutoff = 3.8
        self.sascore_cutoff= 5.2
        
        # File handling for SLATM calculation
        self.sizeof_slatm=50518
        self.mbtypes_path="/home/student7/LucaSchaufelberger/MasterThesis/FORMED_ML/predict/data_test/mbtypes.npy"
        
        # DFT scripts path
        self.utils_path = self.config_dir / "utils"
        self.run_gaussian_relaxation = self.utils_path / "gaussian_GS_relaxation_lowmem.sh"
        self.run_gaussian_log2xyz = self.utils_path / "gaussian_log_to_xyz.sh"
        self.run_gaussian_tddft = self.utils_path / "gaussian_for_gauss_sum.sh"
        self.path_dens_ana_in = self.utils_path / "dens_ana.in"
        

        # 
        self.max_nr_hours_relaxation = 6
        self.max_nr_hours_conversion = 1
        self.max_nr_hours_tddft =4
        
        
        # Model paths for SMILES model
        self.nr_smiles_models=2
        self.smiles_models_location = self.config_dir / "models/smiles_model/"
        self.S1_model_paths=[self.smiles_models_location / "S1_model_1", 
                        self.smiles_models_location / "S1_model_2"  ]
        self.T1_model_paths= [self.smiles_models_location / "T1_model_1", 
                         self.smiles_models_location / "T1_model_2"  ]
        self.S1ehdist_model_paths=[self.smiles_models_location / "S1excd_model_1", 
                              self.smiles_models_location / "S1excd_model_1"  ] 
                

        
        # Model paths for SLATM model
        self.slatm_models_location = self.config_dir / "models/slatm_model/"
        self.S1_slatm_model_path = self.slatm_models_location / "S1_exc_model.sav"
        self.T1_slatm_model_path = self.slatm_models_location / "T1_exc_model.sav"
        self.S1ehdist_slatm_model_path = self.slatm_models_location / "S1_ehdist_model.sav"
        

        
        # Model evaluation parameters
        self.batch_size =16  #2 #For chemberta batch evaluation (better performance), needs to divide the population 
        assert self.pop_size % self.batch_size == 0 
        

