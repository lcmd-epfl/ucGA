import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use the 'Agg' backend for matplotlib to save plots without showing them
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
import xgboost

# Set seaborn style for plots
sns.set()

def load_data(cycle_index, df_index):
    """
    Load DFT results and representations for the specified cycle.
    
    Parameters:
    - cycle_index: The cycle number to load data for.
    - df_index: Column index for the property of interest in the DFT results.
    
    Returns:
    - X_pool: Numpy array of representations.
    - y_pool: Numpy array of property values corresponding to the representations.
    """
    wd = os.getcwd()
    df_DFT = pd.read_csv(f"{wd}/DFT_results_{cycle_index}", header=None)
    df_DFT = df_DFT.drop_duplicates(subset=[0]).dropna()
    
    X_pool = np.load(f"{wd}/DFT_calculations_active/repr_cycle{cycle_index}.npy")
    y_pool = np.array(df_DFT[df_index])
    
    return X_pool, y_pool

def calculate_feature_weights():
    """
    Calculate feature weights for SLATM representations.
    
    Returns:
    - fws: Numpy array containing the feature weights.
    """
    # Define feature weight calculations for different types of potentials
    fw_one_b = 1 / (3 * 13)
    fw_two_b = 1 / (3 * 4368)
    fw_three_b = 1 / (3 * 46137)
    
    fws = np.zeros((50518))
    for i in range(50518):
        if i < 13:
            fws[i] = fw_one_b
        elif i < 4381:
            fws[i] = fw_two_b
        else:
            fws[i] = fw_three_b
            
    assert np.isclose(np.sum(fws), 1), "Sum of feature weights should be 1"
    
    return fws

def retrain_model(X_pool, y_pool, fws, old_model_path):
    """
    Retrain the SLATM model with given data and feature weights.
    
    Parameters:
    - X_pool: Numpy array of representations.
    - y_pool: Numpy array of property values.
    - fws: Feature weights for the XGBoost model.
    - old_model_path: Path to the old model for initialization.
    
    Returns:
    - model: The retrained XGBoost model.
    """
    model = XGBRegressor(
        n_estimators=5000,
        eta=0.05,
        colsample_bytree=0.75,
        max_depth=8,
        eval_metric="mae",
    )
    
    old_model = pickle.load(open(old_model_path, "rb"))
    model.fit(X_pool, y_pool, feature_weights=fws, xgb_model=old_model)
    
    return model

def main():
    """
    Main function to handle the workflow of retraining the SLATM model.
    """
    print("SLATM RETRAINING")
    wd = os.getcwd()
    
    # Parse command-line arguments
    cycle_index = int(sys.argv[1])
    property_to_retrain = sys.argv[2]
    model_index = sys.argv[3]
    
    # Mapping property to retrain to dataframe index and colsample_bytree value
    properties = {
        "S1": {"df_index": 3, "colsample_bytree_value": 0.75},
        "T1": {"df_index": 2, "colsample_bytree_value": 0.25},
        "S1ehdist": {"df_index": 5, "colsample_bytree_value": 0.75}
    }
    df_index = properties[property_to_retrain]["df_index"]
    
    # Determine the path to the old model
    if cycle_index == 1:
        old_model_paths = {
            "S1": "/home/student7/LucaSchaufelberger/MasterThesis/FORMED_ML/models/S1_exc_model.sav",
            "T1": "/home/student7/LucaSchaufelberger/MasterThesis/FORMED_ML/models/T1_exc_model.sav",
            "S1ehdist": "/home/laplaza/Projects/terry_xgb/FORMED_ML/models/S1_ehdist_model.sav"
        }
        old_model_path = old_model_paths[property_to_retrain]
    else:
        old_model=wd+"/models/"+str(property_to_retrain)+"_"+str(model_index)+"_SLATM_retrained_"+str(cycle_index-1)+".sav"

    # Load representations, labels and values
    X_pool,y_pool = load_data(cycle_index, df_index)

    print("Shape of data: ", X_pool.shape, y_pool.shape)  
    fws=calculate_feature_weights()
    
    old_model= pickle.load(open(old_model_path, "rb"))

    # Define the model
    model = XGBRegressor(
        n_estimators=5000,
        eta=0.05,
        colsample_bytree=0.75,
        max_depth=8,
        eval_metric="mae",
)    
    model.fit(
    X_pool, y_pool, feature_weights=fws, xgb_model=old_model
)
    print("finished fitting SLATM",flush=True)


    pickle.dump(model, open(wd+"/models/"+str(property_to_retrain)+"_"+str(model_index)+"_SLATM_retrained_"+str(cycle_index)+".sav", "wb"))
    print("pickled SLATM model",flush=True)

if __name__ == "__main__":
    main()
    
"""
import pickle
#### NOT WORKING, NEED TO UPDATE
import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBRFRegressor
import xgboost
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
sns.set()
wd=os.getcwd()

print("SLATM RETRAINING")

cycle_index=int(sys.argv[1])
property_to_retrain = sys.argv[2]
model_index=sys.argv[3]

if property_to_retrain=="S1": df_index=3
elif property_to_retrain=="T1": df_index=2
elif property_to_retrain=="S1ehdist": df_index=5

if property_to_retrain=="S1": colsample_bytree_value=0.75
elif property_to_retrain=="T1": colsample_bytree_value=0.25
elif property_to_retrain=="S1ehdist": colsample_bytree_value=0.75

if cycle_index==1:    
    if property_to_retrain=="S1": old_model_path="/home/student7/LucaSchaufelberger/MasterThesis/FORMED_ML/models/S1_exc_model.sav"
    elif property_to_retrain=="T1": old_model_path="/home/student7/LucaSchaufelberger/MasterThesis/FORMED_ML/models/T1_exc_model.sav"
    elif property_to_retrain=="S1ehdist": old_model_path="/home/laplaza/Projects/terry_xgb/FORMED_ML/models/S1_ehdist_model.sav"

else:
    old_model_path=wd+"/models/"+str(property_to_retrain)+"_"+str(model_index)+"_SLATM_retrained_"+str(cycle_index-1)+".sav"  
    
    
wd=os.getcwd()
df_DFT=pd.read_csv(wd+"/DFT_results_"+str(cycle_index),header=None)
df_DFT=df_DFT.drop_duplicates(subset=[0])
df_DFT=df_DFT.dropna()


# Load representations, labels and values
X_pool = np.load(wd+"/DFT_calculations_active/repr_cycle"+str(cycle_index)+".npy")
y_pool = np.array(df_DFT[df_index])

print("Shape of data: ", X_pool.shape, y_pool.shape)


# 13 one body pot, 91 2 body pot and 1183 3 body pots.
# 2 body pots are 48 long, 3 body pots are 39 long
# for a total of 50518 SLATM terms
ic = []
one_b = list(range(0, 13))
ic.append(one_b)
two_b = list(range(13, 4368 + 13))
ic.append(two_b)
three_b = list(range(4368 + 13, 46137 + 4368 + 13))
ic.append(three_b)

fw_one_b = 1 / (3 * 13)
fw_two_b = 1 / (3 * 4368)
fw_three_b = 1 / (3 * 46137)
fws = np.zeros((50518))
for i in range(50518):
    if i < 13:
        fws[i] = fw_one_b
    elif i < 4381:
        fws[i] = fw_two_b
    else:
        fws[i] = fw_three_b
assert np.isclose(np.sum(fws), 1)

# Define the model
model = XGBRegressor(
    n_estimators=5000,
    eta=0.05,
    colsample_bytree=0.75,
    max_depth=8,
    eval_metric="mae",
)

print("old_model_path",old_model_path,flush=True)

#old_model= xgboost.Booster(old_model_path)
old_model= pickle.load(open(old_model_path, "rb"))

# Fit the model
model.fit(
    X_pool, y_pool, feature_weights=fws, xgb_model=old_model
)
print("finished fitting SLATM",flush=True)


pickle.dump(model, open(wd+"/models/"+str(property_to_retrain)+"_"+str(model_index)+"_SLATM_retrained_"+str(cycle_index)+".sav", "wb"))
print("pickled SLATM model",flush=True)
"""