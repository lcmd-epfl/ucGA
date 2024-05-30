import numpy as np
from rdkit import Chem
from simpletransformers.classification import ClassificationModel, ClassificationArgs

class ChembertaBatchPredictor:
    def __init__(self, config):
        self.config = config
        self.S1_models = self.load_models(self.config.S1_model_paths)
        self.T1_models = self.load_models(self.config.T1_model_paths)
        self.S1ehdist_models = self.load_models(self.config.S1ehdist_model_paths)
        self.additional_amplification_per_model = 1

    def load_models(self, list_model_paths):
        return [self.load_model(path) for path in list_model_paths]

    @staticmethod
    def load_model(model_path, use_cuda=False):
        model_args = ClassificationArgs()
        model_args.regression = True
        model_args.no_save = False

        return ClassificationModel('roberta', model_path, use_cuda=use_cuda, num_labels=1, args=model_args)

    @staticmethod
    def generate_smile_variations(list_chromosomes, array_shape):
        smiles_array = np.empty(array_shape, dtype=object)
        smiles_array[0, :] = list_chromosomes

        for smile_idx, original_smile in enumerate(list_chromosomes):
            mol = Chem.MolFromSmiles(original_smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {original_smile}")
            for i in range(1, array_shape[0]):
                smiles_array[i, smile_idx] = Chem.MolToSmiles(mol, doRandom=True)

        return smiles_array

    @staticmethod
    def predict_property(model, smiles_subset):
        print(model)
        return np.array(model.predict(smiles_subset.tolist())[0])



    def get_chemberta_output_dict(self, list_chromosomes):
        array_first_axis_shape = self.config.nr_evals_SMILES
        
        smiles_array = self.generate_smile_variations(list_chromosomes, (array_first_axis_shape, len(list_chromosomes)))
        assert len(list_chromosomes) % self.config.batch_size == 0, "List of chromosomes must be divisible by batch size."
        assert len(list_chromosomes) == self.config.pop_size

        array_shape = (self.config.nr_evals_SMILES, self.config.pop_size)
        num_batches = len(list_chromosomes) // self.config.batch_size

        # Initialize arrays in a dictionary
        dict_chemberta = {
            'S1_chemberta': np.zeros(array_shape),
            'T1_chemberta': np.zeros(array_shape),
            'S1ehdist_chemberta': np.zeros(array_shape)
        }

        for batch_index in range(num_batches):
            for model_idx in range(self.config.nr_smiles_models):
                for i in range(self.additional_amplification_per_model + 1):  # Assuming '2' is the amplification factor
                    array_first_idx = model_idx * 2 + i
                    batch_slice = slice(self.config.batch_size * batch_index, self.config.batch_size * (batch_index + 1))

                    # Predictions for each model
                    dict_chemberta['S1_chemberta'][array_first_idx, batch_slice] = self.S1_models[model_idx].predict(smiles_array[array_first_idx, batch_slice].tolist())[0]
                    dict_chemberta['S1ehdist_chemberta'][array_first_idx, batch_slice] = self.S1ehdist_models[model_idx].predict(smiles_array[array_first_idx, batch_slice].tolist())[0]
                    dict_chemberta['T1_chemberta'][array_first_idx, batch_slice] = self.T1_models[model_idx].predict(smiles_array[array_first_idx, batch_slice].tolist())[0]

        return dict_chemberta
    



"""
ARCHIVE,  TO CLEAN UP



import numpy as np
from rdkit import Chem
from simpletransformers.classification import ClassificationModel, ClassificationArgs

class ChembertaBatchPredictor:
    def __init__(self, config):
        self.config = config
        self.S1_models = self.load_models(self.config.S1_model_paths)
        self.T1_models = self.load_models(self.config.T1_model_paths)
        self.S1ehdist_models = self.load_models(self.config.S1ehdist_model_paths)
        self.additional_amplification_per_model =1

    def load_models(self,list_model_paths):
        return [self.load_model(list_model_paths[0]),self.load_model(list_model_paths[1])]
    
    @staticmethod
    def load_model(model_path, use_cuda=False):
        model_args = ClassificationArgs()
        model_args.regression = True
        model_args.no_save = False

        return ClassificationModel('roberta', model_path, use_cuda=use_cuda, num_labels=1, args=model_args)

    @staticmethod
    def generate_smile_variations(list_chromosomes, array_shape):
        smiles_array = np.empty(array_shape, dtype=object)
        smiles_array[0, :] = list_chromosomes

        for smile_idx, original_smile in enumerate(list_chromosomes):
            for i in range(1, array_shape[0]):
                smiles_array[i, smile_idx] = Chem.MolToSmiles(Chem.MolFromSmiles(original_smile), doRandom=True)

        return smiles_array

    @staticmethod
    def predict_property(models, smiles_subset):
        print(models, smiles_subset)
        predictions = [model.predict(smiles_subset.tolist())[0] for model in models]
        return np.array(predictions)


    def get_chemberta_output_dict(self, list_chromosomes):
        array_first_axis_shape = self.config.nr_evals_SMILES

        smiles_array = self.generate_smile_variations(list_chromosomes, (array_first_axis_shape, len(list_chromosomes)))

        assert(len(list_chromosomes) % self.config.batch_size == 0)

        
        # Batch processing of the chromosomes
        S1_array, T1_array, S1ehdist_array = [], [], []
        for batch_index in range(len(list_chromosomes) // self.config.batch_size):
            for first_idx in range(array_first_axis_shape):
                model_idx = first_idx%2
                start_idx = batch_index * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                S1_batch = self.predict_property(self.S1_models, smiles_array[first_idx, start_idx:end_idx])
                T1_batch = self.predict_property(self.T1_models, smiles_array[first_idx, start_idx:end_idx])
                S1ehdist_batch = self.predict_property(self.S1ehdist_models, smiles_array[first_idx, start_idx:end_idx])

                S1_array.append(S1_batch)
                T1_array.append(T1_batch)
                S1ehdist_array.append(S1ehdist_batch)

        # Flatten arrays and aggregate results
        S1_array = np.concatenate(S1_array, axis=1)
        T1_array = np.concatenate(T1_array, axis=1)
        S1ehdist_array = np.concatenate(S1ehdist_array, axis=1)

        return [S1_array, T1_array, S1ehdist_array]

# Example usage:
# predictor = ChembertaPredictor(S1_models, T1_models, S1ehdist_models, batch_size)
# result_dict = predictor.get_chemberta_output_dict(list_chromosomes)
"""
