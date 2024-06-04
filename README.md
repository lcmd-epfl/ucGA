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
    conda env create -f ucga.yml
    conda activate ucGA
    ```
3. **Install [NaviCatGA](https://github.com/lcmd-epfl/NaviCatGA) and [QML](https://github.com/qmlcode/qml) by following the instructions on the respective project websites**

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

Luca Schaufelberger, et al. TODO

## Authors and Contributions

### Contact

For questions, please contact `schaluca@ethz.ch'

## Abstract

Singlet fission has shown potential for boosting the power conversion efficiency of solar cells, but the scarcity of suitable molecular materials hinders its implementation. We introduce an uncertainty-controlled genetic algorithm (ucGA) that concurrently optimizes excited state energies, synthesizability, and singlet exciton size for the discovery of singlet fission materials. These properties are obtained using ensemble machine learning predictions from different molecular representations. We show that uncertainty in the ensemble model predictions can be used to control how far the genetic optimization moves away from previously known molecules. Running the ucGA in an exploitative setup performs local optimization on variations of known singlet fission scaffolds, such as acenes. In an explorative mode, hitherto unknown candidates are generated which display excellent excited state properties for singlet fission. We identify a class of heteroatom-rich mesoionic compounds as acceptors for charge-transfer mediated singlet fission. When included in larger conjugated donor-acceptor systems, these units exhibit strong localization of the triplet state, favorable diradicaloid character and suitable triplet energies for exciton injection into semiconductor solar cells. As the proposed candidates are composed of fragments from synthesized molecules, they are likely synthetically accessible.

---


ARCHIVE
- config.py: Configures the relevant parameters for the genetic optimization and defines the paths. Edit the config.py file to specify paths for input data, models, and output directories as per your system's directory structure.

- main.py: Main file for launching the uncertainty-aware optimization.

- optimization.py: Executes the uncertainty-aware optimization based on NaviCatGA

Folders:
- active_learning: Code used for retraining the models in the active-learning GA setting (see Supporting Information, not necessary for the normal implementation)
- assembler: Automatically assembled the molecules from a given set of fragments
- fitness_evaluation: implementation of the uncertainty-aware fitness function
- model_predictors: Code to predict properties with the SLATM models
- models: Contains the models used for the optimization
- objective_scores: Contains the implementation of the synthetic difficulty and energy score
- quantum_calculations: Code to automatically perform xTB and TD-DFT calculations
- utils


SCScore model can be downloaded
NaviCat GA~  



- adapt paths to models in config


main.py        
