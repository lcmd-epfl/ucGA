# Inverse Design of Singlet Fission Materials with Uncertainty-Controlled Genetic Optimization

![Overview](/Overview.png)

## Overview

This repository contains the implementation of the uncertainty-controlled genetic algorithm (ucGA) for the design of singlet fission materials presented in the paper (TODO:Link).


## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/lucaschaufelberger/ucGA.git
    cd ucGA
    ```

2. **Create and activate the Conda environment**:
    ```bash
    conda env create -f env.yml
    conda activate ucGA
    ```
3. **Further Installations**
    Install [NaviCatGA](https://github.com/lcmd-epfl/NaviCatGA), [theodore](https://github.com/felixplasser/theodore-qc) and [QML](https://github.com/qmlcode/qml) by following the instructions on the respective project websites.

4. **SCScore**: 
   Install the [SCScore](https://github.com/connorcoley/scscore) to utils/scscore.

## Usage

1. **Configure the parameters**:
   Edit `config.py` to set the appropriate paths for input data, models, and output directories.

2. **Run the main optimization**:
    ```bash
    python main.py <output_folder>
    ```
## Important Files

- **config.py**: Configuration file for setting paths and parameters for the genetic optimization. Edit this file to specify paths for input data, models, and output directories according to your system's directory structure.
- **main.py**: Main script for launching the uncertainty-aware optimization.

## Repository Structure


- **assembler/**: Automatically assembling molecules from the reFORMED database.
- **fitness_evaluation/**: Uncertainty-aware fitness function determination.
- **model_predictors/**: Prediction of properties using SLATM models.
- **objective_scores/**: Calculate scores to assess singlet fission propensity.
- **quantum_calculations/**: Scripts for performing xTB and TD-DFT calculations.
- **utils/**: Utility scripts and functions.




## Citation

If you use this code, please cite the following paper:

Luca Schaufelberger,  TODO


### Contact

For questions, please contact `schaluca@ethz.ch'
