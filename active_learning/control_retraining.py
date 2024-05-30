
import time
import os
import pandas as pd
import subprocess
import logging
from pathlib import Path


from UncAGA.quantum_calculations.xtb import XTB_Processor
from UncAGA.quantum_calculations.slatm import generate_SLATM_from_list

class ModelRetrainer:
    def __init__(self, cycle_index, config):
        self.cycle_index = cycle_index
        self.config = config
        
  

    def launch_retraining(self):
        """Launch the retraining process for specified models."""
        print(f"Starting launching, old chemberta paths for cycle {self.cycle_index} are prepared.")

        #load DFT data
        df_DFT = pd.read_csv(self.config.output_path / f"DFT_results_{self.cycle_index}", header=None)
        df_DFT = df_DFT.drop_duplicates(subset=[0]).dropna()
        chrom_hashes = df_DFT[1].tolist()

        # Generate SLATMs
        xyz_files = [self.config.structures_path / f"{chrom_hash}_pysis_conv.xyz" for chrom_hash in chrom_hashes]
        generate_SLATM_from_list(str(self.config.output_path / "DFT_calculations_active/"),
                                 50518,
                                 xyz_files,
                                 appendix='_cycle'+str(self.cycle_index))  
        # Model retraining commands
        
        self.load_model_paths()
        
        self.run_chemberta_retraining()
        self.run_SLATM_retraining()
        
        self.adjust_paths()

        # Final checks and operations
        self.check_SLATM_retraining_completion()
        self.check_chemberta_retraining_completion()

    def load_model_paths(self):
            #print("self.S1_model_paths|",self.config.S1_model_paths)
            self.chemberta_model_paths = [{'type':'S1','path':self.config.S1_model_paths[0],'index':1},
                                          {'type':'S1','path':self.config.S1_model_paths[1],'index':2},
                                          {'type':'T1','path':self.config.T1_model_paths[0],'index':1},
                                          {'type':'T1','path':self.config.T1_model_paths[1],'index':2},
                                          {'type':'S1ehdist','path':self.config.S1ehdist_model_paths[0],'index':1},
                                          {'type':'S1ehdist','path':self.config.S1ehdist_model_paths[1],'index':2}]




    def run_chemberta_retraining(self):
        """Run the ChemBERTa model retraining."""
        for model in self.chemberta_model_paths:
            model_type = model['type']
            path=model['path']
            model_index = model['index']
            

            retraining = subprocess.run(['sh', 
                                         self.config.launch_chemberta_retrain_sh, 
                                         self.config.launch_chemberta_retrain_py, 
                                         str(self.cycle_index), 
                                         str(model_type), 
                                         str(path), 
                                         str(model_index),
                                         str(self.config.output_path)],
                                         cwd=self.config.output_path
                                         )
            logging.info(f"Retraining process initiated for {path}.")

    def run_SLATM_retraining(self):
        """Run the SLATM XGBoost model retraining."""
        for model_type in ["S1","T1","S1ehdist"]:
            print("SLATM LAUNCH")
            retraining = subprocess.run(['sh', self.config.launch_SLATM_retrain_sh, 
                                         self.config.launch_SLATM_retrain_py, 
                                         str(self.cycle_index), 
                                         model_type,
                                         "1"])
            logging.info(f"Retraining process initiated for {model_type}.")
            
            
    def find_checkpoint_folder_for_epoch_10(self,directory):
        """
        Searches for a folder within a given directory that contains a checkpoint for epoch 10.
        This folder must include a specific set of files to be considered a match.

        Parameters:
        - directory (str): The base directory to start the search from.

        Returns:
        - str or None: The path to the matching folder if found, otherwise None.
        """
        logging.info("Starting model finder")
        for root, dirs, _ in os.walk(directory):
            if "checkpoint" in root and "epoch-10" in root:
                logging.info("Found matching folder")
                if self.check_required_files_present(root):
                    return root
                else:
                    return None
        return None

    def check_required_files_present(self,folder_path):
        """
        Checks if all necessary files are present in a given folder.

        Parameters:
        - folder_path (str): The path to the folder to check.

        Returns:
        - bool: True if all required files are present, False otherwise.
        """
        required_filenames = {
            'config.json', 'pytorch_model.bin', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.json', 'merges.txt',
            'tokenizer.json', 'training_args.bin', 'optimizer.pt', 'model_args.json'
        }
        files_present = set(os.listdir(folder_path))
        return required_filenames.issubset(files_present)
    

    def check_SLATM_retraining_completion(self, timeout_hours=4):
        """Check if SLATM model retraining has completed successfully.

        """
        logging.info("Starting time loop to check SLATM model retraining completion.")
        start_time = time.time()
        timeout = timeout_hours * 3600  # Convert hours to seconds

        while time.time() - start_time < timeout:
            all_models_retrained = True
            #for model_name, model_path in self.chemberta_model_paths.items():
            for model_type in ["S1","T1","S1ehdist"]:
                SLATM_model = self.config.output_path / "models" / (str(model_type) + "_1_SLATM_retrained_" + str(self.cycle_index) + ".sav")
                print(SLATM_model)

                if not os.path.exists(SLATM_model):
                    logging.info(f"SLATM Model {model_type} is not yet retrained. Waiting...")
                    all_models_retrained = False
                    break  # Exit the loop to wait more
            if all_models_retrained:
                logging.info("All SLATM models have been successfully retrained.")
                return
            else:
                time.sleep(20)  # Wait before checking again

        logging.warning("Timeout reached. Some models may not have been retrained successfully.")


    def check_chemberta_retraining_completion(self, timeout_hours=4):
        """Check if ChemBERTa model retraining has completed successfully.

        """
        logging.info("Starting time loop to check ChemBERTa model retraining completion.")
        start_time = time.time()
        timeout = timeout_hours * 3600  # Convert hours to seconds

        while time.time() - start_time < timeout:
            all_models_retrained = True
            #for model_name, model_path in self.chemberta_model_paths.items():
            for model in self.chemberta_model_paths:
                model_type = model['type']
                model_index = model['index']
                model_path=Path(f"{self.config.output_path}/models/{model_type}_{model_index}_chemberta_retrained_{self.cycle_index}")
                finished_model_path = self.find_checkpoint_folder_for_epoch_10(model_path)
                final_model_path = model_path / "final"
                if finished_model_path is not None:
                    print("moving now!!!",finished_model_path,final_model_path)
                    subprocess.run(['mv',finished_model_path,final_model_path])
                
                
                
                if not final_model_path.exists():
                    print(f"Model {model_type, model_path} is not yet retrained. Waiting...")
                    logging.info(f"Model {model_type} is not yet retrained. Waiting...")
                    all_models_retrained = False
                    break  # Exit the loop to wait more
            if all_models_retrained:
                logging.info("All models have been successfully retrained.")
                return
            else:
                time.sleep(20)  # Wait before checking again

        logging.warning("Timeout reached. Some models may not have been retrained successfully.")

        
    def adjust_paths(self):
  
        #TODO
        self.config.smiles_models_location = self.config.output_path / "models"
        self.config.S1_model_paths=[self.config.smiles_models_location / f"S1_1_chemberta_retrained_{self.cycle_index}", 
                        self.config.smiles_models_location / f"S1_2_chemberta_retrained_{self.cycle_index}"  ]
        self.config.T1_model_paths= [self.config.smiles_models_location / f"T1_1_chemberta_retrained_{self.cycle_index}", 
                         self.config.smiles_models_location / f"T1_2_chemberta_retrained_{self.cycle_index}"  ]
        self.config.S1ehdist_model_paths=[self.config.smiles_models_location / f"S1ehdist_1_chemberta_retrained_{self.cycle_index}", 
                              self.config.smiles_models_location / f"S1ehdist_2_chemberta_retrained_{self.cycle_index}"  ] 
                
        print("NEW S1 chemberta PATHS",self.config.S1_model_paths)
        
        # Model paths for SLATM model
        self.config.slatm_models_location = self.config.output_path / "models"
        self.config.S1_slatm_model_path = self.config.slatm_models_location / f"S1_SLATM_1_retrained_{self.cycle_index}.sav"
        self.config.T1_slatm_model_path = self.config.slatm_models_location / f"T1_SLATM_1_retrained_{self.cycle_index}.sav"
        self.config.S1ehdist_slatm_model_path = self.config.slatm_models_location / f"S1_1_SLATM_retrained_{self.cycle_index}.sav"
        
        print("NEW S1ehdist slatm PATH",self.config.S1ehdist_slatm_model_path)