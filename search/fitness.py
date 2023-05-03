from commons.genetics import Population, Individual
from commons.utils import init_model
from typing import Iterable, Callable
from metrics.naswot import compute_naswot as naswot
from metrics.logsynflow import compute_logsynflow as logsynflow
from metrics.skipped_layers import compute_skipped_layers as skipped_layers
import numpy as np
import torch
from typing import Tuple

"""
TODO: Fix columns reader - might use a dictionary initialized in metrics.__init__.py with the name
of the scores as key and the corresponding column index in the cachedmetrics directory.
TODO: Eventually, make all scores function one function only with the same inputs but able to differentiate
between the different scores simply using the string `score`.
"""

def score_naswot(individual:Individual,
                 images:torch.Tensor,
                 lookup_table:np.ndarray=None,
                 n_inits:int=3) -> Individual: 
    """Scores each individual with respect to the naswot score"""
    if not hasattr(individual, "naswot_score"):     
        if lookup_table is not None:
            individual.naswot_score = lookup_table[individual.index, 1]
        else:
            # models are initialez with different weights at computing time.
            models = [init_model(individual.net) for _ in range(n_inits)]
            individual.naswot_score = np.mean([naswot(net, inputs=images) for net in models])

    return individual

def score_logsynflow(individual:Individual,
                 images:torch.Tensor,
                 lookup_table:np.ndarray=None,
                 n_inits:int=3) -> Individual:
    """Scores each individual with respect to the log-synflow score"""
    if not hasattr(individual, "logsynflow_score"): 
        if lookup_table is not None:
            individual.logsynflow_score = lookup_table[individual.index, 2]
        else:
            # models are initialez with different weights at computing time.
            models = [init_model(individual.net) for _ in range(n_inits)]
            individual.logsynflow_score = np.mean([logsynflow(net, inputs=images) for net in models])
    
    return individual

def score_skipped(individual:Individual, lookup_table:np.ndarray=None) -> Individual:
    """Scores each individual with the fraction of skipped layers over the possible skip connections"""
    if not hasattr(individual, "skip_score"): 
        if lookup_table is not None:
            individual.skip_score = lookup_table[individual.index, 3]
        else:
            individual.skip_score = skipped_layers(individual.genotype)
    return individual

def score_population(population:Population, scores:Iterable[Callable]): 
    """This function score individuals based on scoring functions in scores"""
    for score_function in scores: 
        population.apply_on_individuals(function=score_function)

def normalize_scores(individual:Individual, population:Population, score:str, style:str="dynamic")->float: 
    """
    Normalizes the scores (stored as class attributes) of each individual according to style
    TODO: Add documentation
    """
    # sanity checks on inputs
    if not isinstance(score, str): 
        raise ValueError(f"Input score '{score}' is not a string!")    
    
    if style in ['minmax', 'dynamic']:
        min_value = getattr(population, f"min_{score}")
        max_value = getattr(population, f"max_{score}")
    elif style == 'standard':        
        mean_value = getattr(population, f"mean_{score}")
        std_value = getattr(population, f"std_{score}")
    else: 
        raise ValueError(f"Style {style} must be one of ['minmax', 'dynamic', 'standard']")

    # mapping score values in the [0,1] range using min-max normalization
    def minmax_individual(individual:Individual):
        """Normalizes in the [0,1] range the value of a given score"""
        current_score = getattr(individual, score)
        if (max_value - min_value) >= 1e-12:
            return (current_score - min_value) / (max_value - min_value)
        else: 
            # only way min and max are equal is that also individual.score is equal to them
            return current_score / max_value if max_value > 1e-6 else current_score
    
    # mapping score values to distribution with mean 0 and std 1
    def standardize_individual(individual:Individual):
        """Normalizes to mean 0 and std 1"""
        current_score = getattr(individual, score)
        if std_value >= 1e-12: 
            return (current_score - mean_value) / std_value
        else:
            return current_score - mean_value
    
    # return normalized score
    return minmax_individual(individual) if style in ["minmax", "dynamic"] else standardize_individual(individual)

def fitness_score(
        individual:Individual,
        population:Population,
        style:str="dynamic",
        weights:Tuple[None, np.array]=None)->float:
    """Sums the three scores to obtain final expression for fitness"""
    scores = ["naswot_score", "logsynflow_score", "skip_score"]  # change here to add more

    # computing the individual's scores
    ind_scores = np.array([normalize_scores(
        individual=individual, 
        population=population, 
        score=score, style=style) 
        for score in scores])
    # getting the weights
    weights_scores = np.ones_like(ind_scores) if weights is None else weights
    # fitness is a convex combinations of score componenents
    weights_scores = weights_scores / weights_scores.sum()
    # fitness is scalar product of weight scores and ind_scores 
    return weights_scores @ ind_scores
