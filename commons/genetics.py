from xautodl.models.cell_infers.tiny_network import TinyNetwork
from torch.nn import Module
from typing import Iterable, Callable, Tuple, List, Union
import numpy as np
from copy import deepcopy as copy
from .nats_interface import NATSInterface
from .utils import *
from itertools import chain

NATSPATH = str(get_project_root()) + "/archive/NATS-tss-v1_0-3ffb9-simple/"

class Individual(): 
    def __init__(
        self, 
        net:TinyNetwork, 
        genotype:list, 
        age:int=0,
        dataset:str="cifar10", 
        searchspace_interface:object=None):
        
        self.scores = {}
        self.net = net
        self._genotype = genotype
        self.age = age

        self._fitness = 0
        self._rank = 0
        # searchspace interface needed to exploit search space properties
        if searchspace_interface is None: 
            self.interface = NATSInterface(path=NATSPATH, dataset=dataset)
        else: 
            self.interface = searchspace_interface
            
    def update_net(self):
        """Over-writes net field in light of genotype"""
        genotype_arch_str = genotype_to_architecture(self.genotype)
        self.net, _ = self.interface.query_with_architecture(architecture_string=genotype_arch_str)

    @property
    def genotype(self): 
        return self._genotype

    def update_genotype(self, new_genotype:List): 
        """Update current genotype with new one. When doing so, also the network field is updated"""
        # sanity check on new genotype
        if not genotype_is_valid(genotype=new_genotype):
            ValueError(f"genotype {new_genotype} is not a valid replacement for {self.genotype}!")

        self._genotype = new_genotype
        self.update_net()

    @property
    def fitness(self): 
        return self._fitness
    
    def update_fitness(self, metric:Callable, attribute:str="net"): 
        """Update the current value of fitness using provided metric"""
        self._fitness = metric(getattr(self, attribute))
    
    def overwrite_fitness(self, new_fitness:float):
        """Overwrite current value of fitness"""
        if isinstance(new_fitness, float) or isinstance(new_fitness, int): 
            self._fitness = new_fitness
        else: 
            raise ValueError(f"New fitness value ({new_fitness}) is not a number!")

    @property
    def rank(self): 
        return self._rank
    
    def update_ranking(self, new_rank:int) -> None: 
        """Updates current ranking of considered architecture"""
        self._rank = new_rank

class Genetic: 
    def __init__(
        self, 
        genome:Iterable[str], 
        strategy:Tuple[str, str]="comma", 
        tournament_size:int=5,
        cross_p:float=0.5):
        
        self.genome = set(genome) if not isinstance(genome, set) else genome
        self.strategy = strategy
        self.tournament_size = tournament_size
        self.cross_probability = cross_p

    def tournament(self, population:Iterable[Individual]) -> Iterable[Individual]:
        """Return tournament, i.e. a random subset of population of size tournament size"""
        return np.random.choice(a=population, size=self.tournament_size).tolist()
    
    def obtain_parents(self, population:Iterable[Individual], n_parents:int=2) -> Iterable[Individual]:
        """Obtain n_parents from population. Parents are defined as the fittest individuals in n_parents tournaments"""
        parents = []
        for p in range(n_parents): 
            tournament = self.tournament(population = population)
            # parents are defined as fittest individuals in tournaments
            parents.append(
                sorted(tournament, key = lambda individual: individual.fitness, reverse=True)[0]
            )
        return parents
    
    def mutate(self, individual:Individual, n_loci:int=1) -> Individual: 
        """Applies mutation to a given individual"""
        mutant_individual = copy(individual)
        for _ in range(n_loci): 
            mutant_genotype = mutant_individual.genotype
            # select a locus in the genotype (that is, where mutation will occurr)
            mutant_locus = np.random.randint(low=0, high=len(mutant_individual.genotype))
            # mapping the locus to the actual gene that will effectively change
            mutant_gene = mutant_genotype[mutant_locus]
            operation, level = mutant_gene.split("~")  # splits the gene into operation and level
            # mutation changes gene, so the current one must be removed from the pool of candidate genes
            mutations = self.genome.difference([operation])
            
            # overwriting the mutant gene with a new one
            mutant_genotype[mutant_locus] = np.random.choice(a=list(mutations)) + f"~{level}"
            mutant_individual.update_genotype(new_genotype=mutant_genotype)
        
        return mutant_individual
    
    def recombine(self, individuals:Iterable[Individual], n_parts:int=2) -> Individual: 
        """Performs recombination of two given `individuals`"""
        if len(individuals) != 2: 
            raise ValueError("Number of individuals cannot be different from 2!")
        
        individual1, individual2 = individuals
        recombinant = copy(individual1)
        
        # select the index in which to cut down the individual
        recombination_locus = np.random.randint(low=0, high=len(individual1.genotype)-1)
        # individual1 is dominant in the recombinant with probability self.cross_p
        realization = np.random.random()
        # defining new genotype of recombinant individual
        if realization < self.cross_probability:
            recombinant_genotype = list(chain(individual1.genotype[:recombination_locus], individual2.genotype[recombination_locus:]))
        else:
            recombinant_genotype = list(chain(individual2.genotype[:recombination_locus], individual1.genotype[recombination_locus:]))
        
        recombinant.update_genotype(list(recombinant_genotype))
        return recombinant

class Population: 
    def __init__(self, space:object, individual:object=Individual, init_population:Union[bool, Iterable]=True): 
        self.space = space
        self.individual = individual
        if init_population:
            self._population = generate_population(searchspace_interface=space, individual=individual)
        else: 
            self._population = init_population
        
        self.oldest = None
        self.worst_n = None
    
    def __iter__(self): 
        for i in self._population: 
            yield i
    
    @property
    def individuals(self):
        return self._population
    
    def update_population(self, new_population:Iterable[Individual]): 
        """Overwrites current population with new one stored in `new_population`"""
        if all([isinstance(el, Individual) for el in new_population]):
            del self._population
            self._population = new_population
        else:
            raise ValueError("new_population is not an Iterable of `Individual` datatype!")

    def fittest_n(self, n:int=1): 
        """Return first `n` individuals based on fitness value"""
        return sorted(self._population, key=lambda individual: individual.fitness, reverse=True)[:n]
    
    def update_ranking(self): 
        """Updates the ranking in the population in light of fitness value"""
        sorted_individuals = sorted(self._population, key=lambda individual: individual.fitness, reverse=True)
        
        # ranking in light of individuals 
        for ranking, individual in enumerate(sorted_individuals):
            individual.update_ranking(new_rank=ranking)

    def update_fitness(self, fitness_function:Callable): 
        """Updates the fitness value of individuals in the population"""
        for individual in self.individuals: 
            individual.overwrite_fitness(fitness_function(individual))
    
    def apply_on_individuals(self, function:Callable, inplace:bool=True)->Union[Iterable, None]: 
        """Applies a function on each individual in the population
        
        Args: 
            function (Callable): function to apply on each individual. Must return an object of class Individual.
            inplace (bool, optional): Whether to apply the function on the individuals in current population or
                                      on a copy of these.
        Returns: 
            Union[Iterable, None]: Iterable when inplace=False represents the individuals with function applied.
                                   None represents the output when inplace=True (hence function is applied on the
                                   actual population.
        """
        modified_individuals = [function(individual) for individual in copy(self._population)]
        if inplace:
            self.update_population(new_population=modified_individuals)
        else:
            return modified_individuals 

    def set_extremes(self, score:str):
        """Set the maximal&minimal value in the population for the score 'score' (must be a class attribute)"""
        # sorting in ascending order
        sorted_population = sorted(self.individuals, key=lambda individual: getattr(individual, score))
        min_value, max_value = getattr(sorted_population[0], score), getattr(sorted_population[-1], score),

        setattr(self, f"max_{score}", max_value)
        setattr(self, f"min_{score}", min_value)


    def normalize_scores(self, score:str, inplace:bool=True)->Union[Iterable, None]: 
        """Normalizes the scores (stored as class attributes) of each individual with respect to the maximal"""
        if not isinstance(score, str): 
            raise ValueError(f"Input score '{score}' is not a string!")

        try:
            min_value, max_value = getattr(self, f"min_{score}"), getattr(self, f"max_{score}")
            # mapping score values in the [0,1] range using min-max normalization
            modified_individuals = copy(self.individuals)

            def minmax_individual(individual:Individual):
                """Normalizes in the [0,1] range the value of a given score"""
                new_individual = copy(individual)
                setattr(
                    new_individual,
                    score, 
                    (getattr(individual, score) - min_value) / (max_value - min_value) if max_value != min_value else 0
                    )
                return new_individual

            # normalizing
            new_population = list(map(
                # mapping each score value in the [0,1] range considering population-wise metrics
                minmax_individual,
                # looping in all individuals
                modified_individuals,     
            ))
            if inplace: 
                self.update_population(new_population=new_population)
            else: 
                return new_population
            
        except AttributeError:  # extremes attribute not present... sorting&setting 
            self.set_extremes(score=score)
            self.normalize_scores(score)
    
    def age(self): 
        """Embeds ageing into the process"""
        def individuals_ageing(individual): 
            individual.age += 1
            return individual

        self.apply_on_individuals(function=individuals_ageing, inplace=True)
    
    def add_to_population(self, new_individuals:Iterable[Individual]): 
        """Add new_individuals to population"""
        self._population = list(chain(self.individuals, new_individuals))
    
    def remove_from_population(self, attribute:str="fitness", n:int=1, ascending:bool=True): 
        """Remove first/last `n` elements from sorted population population in `ascending/descending`
        order based on the value of `attribute`"""
        
        if not all([hasattr(el, attribute) for el in self.individuals]):
            raise ValueError(f"Attribute '{attribute}' is not an attribute of all the individuals!")
        # sort the population based on the value of attribute
        sorted_population = sorted(self.individuals, key=lambda ind: getattr(ind, attribute), reverse=False if ascending else True)
        
        # new population is old population minus the `n` worst individuals with respect to `attribute`
        self.update_population(sorted_population[n:])

    def update_oldest(self, candidate:Individual): 
        """Updates oldest individual in the population"""
        if candidate.age >= self.oldest.age: 
            self.oldest = candidate
        else: 
            pass

    def update_worst_n(self, candidate:Individual, attribute:str="fitness", n:int=2): 
        """Updates worst_n elements in the population"""
        if hasattr(candidate, attribute): 
            if any([getattr(candidate, attribute) < getattr(worst, attribute) for worst in self.worst_n]):
                # candidate is worse than one of the worst individuals
                bad_individuals = self.worst_n + candidate
                # sort in increasing values of fitness
                bad_sorted = sorted(bad_individuals, lambda ind: getattr(ind, attribute))
                self.worst_n = bad_sorted[:n]  # return new worst individuals
    
    def set_oldest(self): 
        """Sets oldest individual in population"""
        self.oldest = max(self.individuals, key=lambda ind: ind.age)
    
    def set_worst_n(self, attribute:str="fitness", n:int=2): 
        """Sets worst n elements based on the value of arbitrary attribute"""
        self.worst_n = sorted(self.individuals, key=lambda ind: getattr(ind, attribute))[:n]
    

def generate_population(searchspace_interface:NATSInterface, individual:Individual)->list: 
    """Generate a population of individuals"""
    # at first generate full architectures and cell-structure
    architectures, cells = searchspace_interface.generate_random_samples(n_samples=20)
    
    # mapping strings to list of genes (~genome)
    genotypes = map(lambda cell: architecture_to_genotype(cell), cells)
    # turn full architecture and cell-structure into genetic population individual
    population = [individual(net=net, genotype=genotype) for net, genotype in zip(architectures, genotypes)]
    return population
