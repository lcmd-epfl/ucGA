import pickle
import numpy as np

class SLATMPredictor:
    def __init__(self, config, filename):
        self.config = config
        self.data_path = self.config.structures_path
        self.filename = filename
        self.X_pool = np.load(f"{self.data_path}/repr{filename}.npy")
        self.names_pool = np.load(f"{self.data_path}/names{filename}.npy")
        self.prediction_type = None

    def run_model(self, model_path):
        ys_all, names_all = self.load_and_predict(model_path)
        return ys_all[0]

    def load_and_predict(self, model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        y_hat = model.predict(self.X_pool)
        return y_hat, self.names_pool






"""
import pickle
import numpy as np

class SLATMPredictor:
    def __init__(self, config,filename):
        self.config = config
        self.data_path=self.config.structures_path
        self.filename=filename
        self.X_pool = np.load(f"{self.data_path}/repr{filename}.npy")
        self.names_pool = np.load(f"{self.data_path}/names{filename}.npy")
        self.prediction_type=None
        
    def run_model(self,model_path):
        ys_all, names_all = self.load_and_predict(model_path, self.data_path)
        #ModelPredictor.write_predictions_to_csv(ys_all, names_all, output_path, prediction_suffix + self.filename)
        return ys_all[0]
    
    def load_and_predict(self,model_path):
        model = pickle.load(open(model_path, "rb"))

        y_hat = model.predict(self.X_pool)
        return y_hat, self.names_pool

'''
    @staticmethod
    def write_predictions_to_csv(predictions, names, output_path):
        with open(f"{output_path}/predictions{self.appendix}.csv", "w") as f:
            print("Filename,Prediction", file=f)
            for name, predicted in zip(names, predictions):
                print(f"{name},{predicted}", file=f)
'''
  
"""