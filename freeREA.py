from commons.nats_interface import NATSInterface
from commons.genetics import Genetic, Population, Individual
from commons.utils import get_project_root, read_lookup_table, read_test_metrics, load_images
from commons.dataset import Dataset
from typing import Iterable, Callable
from tqdm import tqdm
from metrics.naswot import compute_naswot as naswot
from metrics.logsynflow import compute_logsynflow as logsynflow
from metrics.skipped_layers import compute_skipped_layers as skipped_layers
import numpy as np

"""TODO: Args for number of generations and number of generations"""

# dataset = Dataset(name="cifar100")
# images = dataset.random_examples()
dataset = "imagenet"
images = load_images(dataset=dataset)

def score_naswot(individual:Individual, lookup_table:np.ndarray=None): 
    """Scores each individual with respect to the naswot score"""
    if not hasattr(individual, "naswot_score"):     
        if lookup_table is not None:
            individual.naswot_score = lookup_table[individual.index, 1]
        else:
            individual.naswot_score = naswot(individual.net, inputs=images)
    return individual

def score_logsynflow(individual:Individual, lookup_table:np.ndarray=None): 
    """Scores each individual with respect to the log-synflow score"""
    if not hasattr(individual, "logsynflow_score"): 
        if lookup_table is not None:
            individual.logsynflow_score = lookup_table[individual.index, 2]
        else:
            individual.logsynflow_score = logsynflow(individual.net, inputs=images)
    return individual

def score_skipped(individual:Individual, lookup_table:np.ndarray=None): 
    """Scores each individual with the fraction of skipped layers over the possible skip connections"""
    if not hasattr(individual, "skip_score"): 
        if lookup_table is not None:
            individual.skip_score = lookup_table[individual.index, 3]
        else:
            individual.skip_score = skipped_layers(individual.genotype)
    return individual

def score_population(population:Population, scores:Iterable[Callable], lookup_table:np.ndarray=None): 
    """This function score individuals based on scoring functions in scores"""
    for score_function in scores: 
        population.apply_on_individuals(function=score_function, lookup_table=lookup_table)

def fitness_score(individual:Individual)->float: 
    """Sums the three scores to obtain final expression for fitness"""
    scores = ["naswot_score", "logsynflow_score", "skip_score"]
    return sum([getattr(individual, score) for score in scores])

def solve(max_generations:int=100, pop_size:int=25, lookup:bool=True): 
    # instantiating a NATSInterface object
    NATS_PATH = str(get_project_root()) + "/archive/NATS-tss-v1_0-3ffb9-simple/"
    nats = NATSInterface(path=NATS_PATH, dataset=dataset)
    # read the lookup table
    lookup_table = read_lookup_table(dataset=dataset) 

    # initialize a random population
    population = Population(space=nats, init_population=True, n_individuals=pop_size)

    scores = [score_naswot, score_logsynflow, score_skipped]
    # scoring the population based on the scoring functions defined.
    # If a lookup table is specified, simply read the lookup table.
    if lookup:
        score_population(population=population, scores=scores, lookup_table = lookup_table)
    else:
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
        if lookup:
            score_population(population=population, scores=scores, lookup_table = lookup_table)
        else:
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

    best_individual = max(population.individuals, key=lambda ind: ind.fitness)

    # retrieve test accuracy for this individual
    test_metrics = read_test_metrics(dataset=dataset)
    test_accuracy = test_metrics[best_individual.index, 1]

    return (best_individual, test_accuracy)

class FreeREA: 
    def __init__(self):
        pass
    
    def search(self, max_generations:int=100, pop_size:int=25): 
        result = solve(max_generations=max_generations, pop_size=pop_size)
        return result

if __name__=="__main__": 
    best_individual, test_accuracy = solve()
    print(f"Best Individual: {best_individual.genotype}, with the following scores:")
    print(f'    "naswot_score = {best_individual.naswot_score}')
    print(f'    "logsynflow_score = {best_individual.logsynflow_score}')
    print(f'    "skip_score = {best_individual.skip_score}')
    print(f'Its test accuracy is: {test_accuracy}')
    print(':)')