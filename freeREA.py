from commons.nats_interface import NATSInterface
from commons.genetics import Genetic, Population, Individual
from commons.utils import get_project_root
from commons.dataset import Dataset
from typing import Iterable, Callable
from tqdm import tqdm
from metrics.naswot import compute_naswot as naswot
from metrics.logsynflow import compute_logsynflow as logsynflow
from metrics.skipped_layers import compute_skipped_layers as skipped_layers
import time

"""TODO: Args for number of generations and number of generations"""

dataset = Dataset(name="cifar10")
images = dataset.random_examples()

def score_naswot(individual:Individual): 
    """Scores each individual with respect to the naswot score"""
    if not hasattr(individual, "naswot_score"): 
        individual.naswot_score = naswot(individual.net, inputs=images)
    return individual

def score_logsynflow(individual:Individual): 
    """Scores each individual with respect to the log-synflow score"""
    if not hasattr(individual, "logsynflow_score"): 
        individual.logsynflow_score = logsynflow(individual.net, inputs=images)
    return individual

def score_skipped(individual:Individual): 
    """Scores each individual with the fraction of skipped layers over the possible skip connections"""
    if not hasattr(individual, "skip_score"): 
        individual.skip_score = skipped_layers(individual.genotype)
    return individual

def score_population(population:Population, scores:Iterable[Callable], lookup:bool=True): 
    """This function score individuals based on scoring functions in scores"""
    for score_function in scores: 
        population.apply_on_individuals(function=score_function, inplace=True)

def fitness_score(individual:Individual)->float: 
    """Sums the three scores to obtain final expression for fitness"""
    scores = ["naswot_score", "logsynflow_score", "skip_score"]
    return sum([getattr(individual, score) for score in scores])

def solve(max_generations:int=100, pop_size:int=25): 
    # instantiating a NATSInterface object
    NATS_PATH = str(get_project_root()) + "/archive/NATS-tss-v1_0-3ffb9-simple/"
    nats = NATSInterface(path=NATS_PATH, dataset="cifar10")

    # initialize a random population
    population = Population(space=nats, init_population=True, n_individuals=pop_size)

    scores = [score_naswot, score_logsynflow, score_skipped]
    # scoring the population based on the scoring functions defined
    score_population(population=population, scores=scores)
    # normalizing scores before computing fitness value
    for score in ["naswot_score", "logsynflow_score", "skip_score"]: 
        population.normalize_scores(score=score, inplace=True)  # normalize values to bring metrics in the same range
    
    # turn score in fitness value
    population.update_fitness(fitness_function=fitness_score)

    # initialize the object taking care of performing genetic operations
    genome = {'none', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect', 'nor_conv_1x1'}
    genetic_operator = Genetic(genome=genome)

    MAX_GENERATIONS = max_generations
    pop = population.individuals

    for gen in tqdm(range(MAX_GENERATIONS)):
        # perform ageing
        population.age()
        # obtain parents
        parents = genetic_operator.obtain_parents(population=pop)
        # obtain recombinant child
        child = genetic_operator.recombine(individuals=parents)
        # mutate parents
        mutant1, mutant2 = [genetic_operator.mutate(parent) for parent in parents]
        # add mutants and child to population
        population.add_to_population([child, mutant1, mutant2])
        # score the new population
        score_population(population=population, scores=scores)
        # normalize scores in the 0-1 range
        for score in ["naswot_score", "logsynflow_score", "skip_score"]: 
            population.set_extremes(score=score)
            population.normalize_scores(score=score, inplace=True)  # normalize values to bring metrics in the same range
        
        # compute fitness value
        population.update_fitness(fitness_score)
        # prune from population worst (from fitness perspective) individuals
        population.remove_from_population(n=2)
        # prune from population oldest individual
        population.remove_from_population(attribute="age", ascending=False)
        # overwrite population
        pop = population.individuals

    return max(population.individuals, key=lambda ind: ind.fitness)

class FreeREA: 
    def __init__(self):
        pass
    
    def search(self, max_generations:int=100, pop_size:int=25): 
        result = solve(max_generations=max_generations, pop_size=pop_size)
        return result

if __name__=="__main__": 
    solve()