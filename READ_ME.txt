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