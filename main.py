import ray
import os
from navicatGA.base_solver import GenAlgSolver
from pathlib import Path

import sys

## TOODOOO Edit this
path=sys.argv[1]

sys.path.append("/home/student7/LucaSchaufelberger/MasterThesis/Paper_Data/")

from UncAGA.fitness_evaluation.fitness_evaluation import calculate_fitness_uncertainty_aware_parallelized
from optimization import UncAGA_Runner
from config import Config
#import constants as const



def main():
    # Set up parallelization
    working_dir = "/home/student7/LucaSchaufelberger/MasterThesis/Paper_Data"
    python_path = os.environ.get('PYTHONPATH', '')
    
    
    # Append your working directory to PYTHONPATH
    python_path += f":{working_dir}"
    #print("PYthonpat",python_path)
    runtime_env = {
        #"working_dir": working_dir,
        "env_vars": {"PYTHONPATH": python_path}
    }
    ray.init(num_cpus=24,runtime_env=runtime_env)
    #ray.get(check_env.remote())


    # Modifications to fitness function determination
    GenAlgSolver.calculate_fitness = calculate_fitness_uncertainty_aware_parallelized
    
    # load configuration class with parameters
    config = Config(path,-2) #exploitative in this case  
    
    # run the uncertainty-aware genetic algorithm (UncAGA)
    uncaga_runner = UncAGA_Runner(config)  
    uncaga_runner.run()  

if __name__ == "__main__":
    main()

    


