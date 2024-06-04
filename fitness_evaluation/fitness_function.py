import hashlib
import logging
import sys
from pathlib import Path
import csv
from rdkit.Chem import RDConfig
import numpy as np
import ray
import os

import sys
sys.path.append("/home/student7/LucaSchaufelberger/MasterThesis/Paper_Data/")
import UncAGA.utils.scscore
import UncAGA.utils.scscore.scscore.standalone_model_numpy as standalone_model_numpy

from UncAGA.model_predictors.slatm_model_eval import SLATMPredictor
from UncAGA.config import Config
from UncAGA.quantum_calculations.xtb import XTB_Processor
from UncAGA.quantum_calculations.slatm import generate_single_slatm
from UncAGA.objective_scores.energy_score import energy_score
from UncAGA.objective_scores.synthetic_difficulty import synthetic_difficulty_score


# Append paths if absolutely necessary (not recommended)
sys.path.append(Path(RDConfig.RDContribDir) / 'SA_Score')

@ray.remote
class FitnessEvaluator:
    def __init__(self, config,list_chromosomes, generation, chemberta_dict):
        """
        - chemberta_dict (dict): Dictionary containing chemberta model predictions.
        - list_chromosomes (list): List of molecular SMILES representations.
        - index (int): Index of the molecule to be evaluated in list_chromosomes
        - generation (int): Current generation in the evolutionary algorithm.
        """
        
        #new_paths="/home/student7/LucaSchaufelberger/MasterThesis/Paper_Data/"
        #os.environ['PYTHONPATH'] += os.pathsep + new_paths
        
        
        import sys
        sys.path.append("/home/student7/LucaSchaufelberger/MasterThesis/Paper_Data/")
        import UncAGA.utils.scscore
        import UncAGA.utils.scscore.scscore.standalone_model_numpy as standalone_model_numpy

        from UncAGA.model_predictors.slatm_model_eval import SLATMPredictor
        from UncAGA.config import Config
        from UncAGA.quantum_calculations.xtb import XTB_Processor
        from UncAGA.quantum_calculations.slatm import generate_single_slatm
        from UncAGA.objective_scores.energy_score import energy_score
        from UncAGA.objective_scores.synthetic_difficulty import synthetic_difficulty_score        
        print("2",sys.path,flush=True)

        self.list_chromosomes = list_chromosomes
        self.generation = generation
        self.chemberta_dict = chemberta_dict
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize default values
        self.energy_score = np.full(self.config.nr_evals_per_chrom, np.nan)
        self.S1 = np.full(self.config.nr_evals_per_chrom, np.nan)
        self.T1 = np.full(self.config.nr_evals_per_chrom, np.nan)
        self.S1ehdistance = np.full(self.config.nr_evals_per_chrom, np.nan)
        self.array_synth_score_average = np.full(self.config.nr_evals_per_chrom, np.nan)

        # Initialize default scores and parameters
        self.synth_score_average = 1
        self.S1ehdistance_score = 0
        self.sco_scs = 5
        self.sco_sa = 10
        self.nr_heavy_atoms = 50
        self.chrom_hash = ''        



    def evaluate(self, index):
        """
        Perform fitness evaluation for a given set of molecules using multiple models.

        Parameters:
        - chemberta_dict (dict): Dictionary containing chemberta model predictions.
        - list_chromosomes (list): List of molecular SMILES representations.
        - index (int): Index of the molecule to be evaluated in list_chromosomes
        - generation (int): Current generation in the evolutionary algorithm.

        Returns:
        list: List containing energy score values, synthetic accessability score, and exciton size.

        The function performs fitness evaluation for a given molecule represented by its SMILES string.
        It combines the predictions from chemberta models, SLATM models, and other scoring functions to
        calculate fitness values. The results are logged, and a list containing fitness values, synth score
        averages, and S1ehdistance scores is returned.

        Note: The function makes use of various external functions and models. Ensure that the required modules and models are
        accessible in the specified paths.

        """
        try:
            self.smiles = self.list_chromosomes[index]
            #Perform xTB Optimization
            xtb_processor = XTB_Processor(self.smiles,self.config)
            self.chrom_hash = xtb_processor.smiles_hash
            xtb_processor.create_xyz_with_xtb()
            xtb_processor.generate_single_slatm()
            
           
            ########## SCORE 1 ##########
            
            #read out chemberta (SMILES model) predictions
            S1_chemberta = list(self.chemberta_dict["S1_chemberta"][:, index])
            T1_chemberta = list(self.chemberta_dict["T1_chemberta"][:, index])
            S1ehdist_chemberta = list(self.chemberta_dict["S1ehdist_chemberta"][:, index])
            
            #perform SLATM model predictions
            predictor = SLATMPredictor(self.config,"{}_pysis_conv".format(xtb_processor.smiles_hash))
            S1_SLATM = predictor.run_model(self.config.S1_slatm_model_path)
            T1_SLATM = predictor.run_model(self.config.T1_slatm_model_path)
          
            self.S1 = S1_chemberta
            self.S1.extend([S1_SLATM,S1_SLATM])
            self.T1 = T1_chemberta
            self.T1.extend([T1_SLATM,T1_SLATM])
            print("S1 and T1", len(self.S1),self.T1)
            
            for i in range(self.config.nr_evals_per_chrom):
                self.energy_score[i]= energy_score(self.T1[i], self.S1[i])
            print(self.S1,self.T1,self.energy_score)
            
            ########## SCORE 2 ##########
            self.array_synth_score_average= synthetic_difficulty_score(self.smiles,self.config)
            print("self.array_synth_score_average",self.array_synth_score_average)
            ########## SCORE 3 ##########
            S1ehdistance_SLATM = predictor.run_model(self.config.S1ehdist_slatm_model_path)
            
            self.S1ehdistance = S1ehdist_chemberta
            self.S1ehdistance.extend([S1ehdistance_SLATM,S1ehdistance_SLATM])



        except Exception as e:
            self.logger.error(f"Error in fitness evaluation for index {index}: {e}")

        # Log properties
        self._log_properties() 

        # Return the evaluation results
        return (self.energy_score, np.full(self.config.nr_evals_per_chrom,self.array_synth_score_average), self.S1ehdistance)
        
    def _log_properties(self):
        """
        Write log data to a CSV file.

        Args:
        path (str): The directory path where the log file will be saved.
        generation, chromosome, chrom_hash, fitness_0, synth_score_average, 
        S1ehdistance_score, S1, T1, sco_scs, sco_sa, nr_heavy_atoms, S1ehdistance: 
        Various parameters to be logged in the file.
        """
        file_name = f'{self.config.output_path}/Fitness_logger_S1_T1.csv'
        data_to_write = [self.generation, self.smiles, self.chrom_hash, self.energy_score, self.synth_score_average, 
                         self.S1ehdistance_score, self.S1, self.T1, self.sco_scs, self.sco_sa, self.nr_heavy_atoms, self.S1ehdistance]
        
        
        print(data_to_write)
        try:
            with open(file_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_write)
        except IOError as e:
            print(f"Error writing to file {file_name}: {e}")

    