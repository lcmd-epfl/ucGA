import numpy as np
import pandas as pd
import ray
import csv
import logging
import numpy as np
import sys
from ucGA.fitness_evaluation.chemberta_batch_prediction import ChembertaBatchPredictor
from ucGA.fitness_evaluation.fitness_function import FitnessEvaluator
from ucGA.assembler.assembler import combine_fragments



def calculate_fitness_uncertainty_aware_parallelized(self, population):
    """Refactored function to calculate fitness in an uncertainty-aware manner"""
    assembled_population = [combine_fragments(chromosome,self.config.connections) for chromosome in population]

    
    performance_values = calculate_performance_values(assembled_population, self.config, self.round_active_eval, self.generations_)
    
    #handle nan values correctly
    scores = process_fitness_data(self,performance_values,self.config.pop_size,self.config.nr_evals_per_chrom)
    
    #calculate confidence bound
    fitness_CB = calculate_confidence_bound(scores,  self.config.lambda_cb)
    
    if self.generations_==self.config.nr_ml_runs_per_iteration:
        log_CB(self.round_active_eval, assembled_population, fitness_CB, self.config.output_path)
    
    return fitness_CB, fitness_CB


def calculate_performance_values(assembled_population,config, round_active_eval, generation):
    sys.path.append("/home/student7/LucaSchaufelberger/MasterThesis/Paper_Data/")
    print(sys.path)

    """Calculate fitness in parallel using Ray."""
    chembertabatchpredictor = ChembertaBatchPredictor(config)
    chemberta_dict = chembertabatchpredictor.get_chemberta_output_dict(assembled_population)
    
    print("chemberta_dict",chemberta_dict)
    
    """
    performance_values=[]
    for i in range(len(assembled_population)):
        fitness_evaluator=FitnessEvaluator(config,assembled_population,generation,chemberta_dict)
        #print(fitness_evaluator.evaluate(i).shape)
        performance_values.append(fitness_evaluator.evaluate(i))
    """
    
    #TODO PARALLELIZE
    fitness_evaluators = [FitnessEvaluator.remote(config,assembled_population,generation,chemberta_dict) for i in range(len(assembled_population))]
    performance_values_remote = [fitness_evaluators[i].evaluate.remote(i) for i in range(len(assembled_population))]
    performance_values = ray.get(performance_values_remote)
    
    print(performance_values)
    return performance_values

def process_fitness_data(self,scores,pop_size,nr_evals_per_chrom):
    """Process raw fitness data, handling NaN values"""
    print('SCOOORES',scores)
    #reshape to prepare for scalarization (vide infra)
    scores = np.swapaxes(np.array(scores), 1, 2)
        
    # the axis now are of population size, the number of evaluations per chromosome, and the number of objectives (i.e. 3)
    assert scores.shape == (pop_size, nr_evals_per_chrom, 3)
    
    # reshape, such that can be scalarized by chimera
    scores_reshaped = scores.reshape(pop_size * nr_evals_per_chrom, 3)
    
    # identify the predictions where one value is nan
    nan_columns = pd.isna(scores_reshaped).any(axis=1)
    
    print(nan_columns)
    
    # remove nan columns for chimera (chimera can't handle nan values)
    array_reshaped_without_nan_columns = scores_reshaped[~nan_columns]

    # fitness: 1 is the maximum
    performance_values = 1 - self.scalarizer.scalarize(array_reshaped_without_nan_columns)
    
    # initialize the performance value array, default is zero (everywhere a nan is detected)
    recovered_array = np.zeros(scores_reshaped.shape[0])
    recovered_array[~nan_columns] = performance_values

    print("recovered with nan handling", recovered_array)
    
    # reshape back to the TODO
    performance_values_reshaped = recovered_array.reshape(pop_size, nr_evals_per_chrom)
    
    print(performance_values_reshaped)
    return performance_values_reshaped

def log_CB(round_active_eval, assembled_population, fitness_CB,output_path):
    """Log Upper Confidence Bound (UCB) calculations."""
    with open(f'{output_path}/UCB/UCB_{round_active_eval}', 'a', newline='') as f:
        writer = csv.writer(f)
        for pop, ucb in zip(assembled_population, fitness_CB):
            writer.writerow([pop, ucb])


def calculate_confidence_bound(performance_values,lambda_cb): 
    print("performance", performance_values.shape)
    fitness_average= np.mean(performance_values,axis=1)
    fitness_std_dev= np.std(performance_values,axis=1)
    fitness_std_dev_chemberta= np.std(performance_values[:,:4],axis=1)
    fitness_std_dev_SLATM= np.std(performance_values[:,4:6],axis=1)
    
    print("std devs", [fitness_std_dev,fitness_std_dev_chemberta,fitness_std_dev_SLATM])
    
    #TODO double check that
    fitness_std_dev_renorm = np.average([fitness_std_dev,fitness_std_dev_chemberta,fitness_std_dev_SLATM],axis=0,weights=[0.6,0.3,0.1])


    fitness_CB = fitness_average + lambda_cb * fitness_std_dev_renorm

    fitness_CB[np.isnan(fitness_CB)] = 0
    return fitness_CB
