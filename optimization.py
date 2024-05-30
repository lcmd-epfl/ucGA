import os
import csv
import logging
import pandas as pd
from chimera import Chimera

from config import Config
from navicatGA.smiles_solver import SmilesGenAlgSolver
from UncAGA.assembler.assembler import combine_fragments
from UncAGA.quantum_calculations.dft import DFTBatchEvaluator

class UncAGA_Runner:
    def __init__(self, config):
        self.config = config
        self.logger = self._prepare_logging_files()

        self.solver = self.initialize_solver()
        self.solver.config = config

    def run(self):
        """
        Run the genetic optimization algorithm using the provided configuration.
        """
        print("output_path",self.config.output_path)
        print(os.path.join(self.config.output_path, 'Fitness_logger.csv'))
        with open(os.path.join(self.config.output_path, 'Fitness_logger.csv'), 'a', newline='') as f:
            print("opened")
                
        iteration_idx = 1
        self._execute_optimization_cycle(iteration_idx)


    def initialize_solver(self):
        """
        Initialize the SmilesGenAlgSolver with parameters from the configuration.
        """
        chimera = Chimera(tolerances=self.config.tolerances, absolutes=self.config.absolutes, goals=["max","min","max"])
        
        solver = SmilesGenAlgSolver(
            n_genes=self.config.n_genes,
            pop_size=self.config.pop_size,
            max_gen=self.config.max_gen,
            alphabet_list=self.config.alphabet_list,
            chromosome_to_smiles=self.assembler_smi,
            fitness_function=self._void_func,
            starting_random=True,
            logger_level="INFO",
            n_crossover_points=1,
            verbose=True,
            prune_duplicates=True,
            progress_bars=False,
            to_file=True,
            to_stdout=True,
            plot_results=True,
            scalarizer=chimera
        )

        solver.round_active_eval = 1
        return solver


   
    def _execute_optimization_cycle(self, iteration_idx):
        """
        Execute a single optimization cycle of the genetic algorithm.
        """

        for generation in range(self.config.nr_ml_runs_per_iteration):
            self.solver.write_UCB_this_generation=False 
            if generation==self.config.nr_ml_runs_per_iteration-1:
                self.solver.write_UCB_this_generation=True
         
            self.solver.solve(1)
            self._log_results()
            
        self._launch_dft()
        #self.solver.round_active_eval += 1

    def _launch_dft(self):
        """
        Launch Density Functional Theory calculations.
        """
        try:
            df_UCB = pd.read_csv(os.path.join(self.config.output_path, "UCB", f"UCB_{self.solver.round_active_eval}"), header=None)
            print(df_UCB)
            dftbatchevaluator = DFTBatchEvaluator(list(df_UCB[0]),self.solver.round_active_eval,self.config)
        except Exception as e:
            self.logger.error(f"Error in DFT launch: {e}")

    def _prepare_logging_files(self):
        """
        Prepare logging files based on the provided path in the configuration.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.config.output_path, 'genetic_algorithm.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def _log_results(self):
        """
        Log the results of the genetic algorithm.
        """
        #try:
        log_data = []
        for chrom in range(self.config.pop_size):
            print("LOGGING",chrom,self.solver.population_)
            assembled_chrom = self.assembler_smi(self.solver.population_[chrom])
            log_data.append([assembled_chrom, self.solver.population_[chrom], self.solver.fitness_[chrom], self.solver.generations_])



        with open(os.path.join(self.config.output_path, 'Fitness_logger.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(log_data)
        #except Exception as e:
        #    self.logger.error(f"Error while logging final results: {e}")

    def assembler_smi(self,chromosome):
        """
        Assemble a molecular SMILES string from fragments.

        Parameters:
        - chromosome (list): List representing the genes (molecular fragments).

        Returns:
        str: Molecular SMILES string assembled from the specified fragments.

        """
        smi_assembl = combine_fragments(chromosome,self.config.connections)
        return smi_assembl

    def _void_func(self):
        pass
 