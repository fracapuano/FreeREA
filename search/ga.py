from commons.nats_interface import NATSInterface
from commons.genetics import Genetic, Population
from commons.utils import get_project_root, read_lookup_table, read_test_metrics, load_images
from .fitness import *
from typing import Callable
from tqdm import tqdm
from typing import Union

FreeREA_dict = {
    "n": 3,  # tournament size
    "N": 5,  # population size
    "mutation_prob": 1.,  # always mutates
    "recombination_prob": 1.,  # always recombines
    "P_parent1": 0.5,  # fraction of child that comes from parent1 on average
    "n_mutations": 1,  # how many loci to mutate
    "loci_prob": None,
}

class GeneticSearch:
    def __init__(self, 
                 dataset:str="cifar10", 
                 lookup:bool=True, 
                 genetics_dict:dict=FreeREA_dict, 
                 init_population:Union[None, Iterable[Individual]]=None):
        
        self.dataset = dataset
        self.score_names = ["naswot_score", "logsynflow_score", "skip_score"]
        self.lookup = lookup
        self.lookup_table = read_lookup_table(dataset=self.dataset) if self.lookup else None 
        self.genetics_dict = genetics_dict
        # get a random batch from dataset
        self.images = load_images(dataset=dataset)

        # instantiating a NATSInterface object
        NATS_PATH = str(get_project_root()) + "/archive/NATS-tss-v1_0-3ffb9-simple/"
        self.nats = NATSInterface(path=NATS_PATH, dataset=self.dataset)

        # instantiating a population
        self.population = Population(
            space=self.nats, 
            init_population=True if init_population is None else init_population, 
            n_individuals=self.genetics_dict["N"],  # N = population size
            normalization="dynamic"
        )

        # initialize the object taking care of performing genetic operations
        self.genetic_operator = Genetic(
            genome=self.nats.nats_ops(), 
            strategy="comma", # population evolution strategy
            tournament_size=self.genetics_dict["n"], 
        )

        # read the lookup table conditionally on using lookup tables in the first place
        self.lookup_table = read_lookup_table(dataset=self.dataset) if self.lookup else None
        # preprocess population
        self.preprocess_population()

    def preprocess_population(self): 
        """
        Applies scoring and fitness function to the whole population. This allows each individual to 
        have the appropriate fields.
        """
        # score the population
        self.score_and_extremes(scores=self.get_metrics())
        # assign the fitness score
        self.assign_fitness()

    def perform_mutation(
            self,
            individual:Individual,
            )->Individual:
        """Performs mutation with respect to genetic ops parameters"""
        realization = np.random.random()
        if realization <= self.genetics_dict["mutation_prob"]:  # do mutation
            mutant = self.genetic_operator.mutate(
                individual=individual, 
                n_loci=self.genetics_dict["n_mutations"], 
                genes_prob=self.genetics_dict["loci_prob"]
            )
            return mutant
        else:  # don't do mutation
            return individual

    def perform_recombination(
            self, 
            parents:Iterable[Individual],
        )->Individual:
        """Performs recombination with respect to genetic ops parameters"""
        realization = np.random.random()
        if realization <= self.genetics_dict["recombination_prob"]:  # do recombination
            child = self.genetic_operator.recombine(
                individuals=parents, 
                P_parent1=self.genetics_dict["P_parent1"]
            )
            return child
        else:  # don't do recombination - simply return 1st parent
            return parents[0]  
    # TODO: Merge compute fitness and assign_fitness
    def score_and_extremes(self, scores:Iterable[Callable]): 
        """This function scores the whole population and sets extremes values for the population."""
        score_population(population=self.population, scores=scores)
        for score in self.score_names:
            self.population.set_extremes(score=score)

    def compute_fitness(self, individual:Individual, population:Population): 
        """This function returns the fitness of individuals according to FreeREA's paper"""
        return fitness_score(individual=individual, population=population, style="dynamic", weights=None)

    def assign_fitness(self):
        """This function assigns to each invidual a given fitness score."""
        # define a fitness function and compute fitness for each individual
        fitness_function = lambda individual: self.compute_fitness(individual=individual,
                                                                   population=self.population
                                                                   )
        self.population.update_fitness(fitness_function=fitness_function)

    def get_metrics(self, dataset:str=None)->Iterable[Callable]: 
        """
        This function returns an iterable of functions instantiated relatively to the current
        dataset and a sample batch.
        """
        # boolean switch
        custom_images = False
        if dataset is not None:  # return metrics for custom dataset
            images = load_images(dataset=dataset)
            custom_images = True
        
        images = self.images if not custom_images else images
        # computing the functions with respect to the different available datasets
        get_naswot = lambda individual: score_naswot(
            individual=individual, images=images, lookup_table=self.lookup_table
            )
        get_logsynflow = lambda individual: score_logsynflow(
            individual=individual, images=images, lookup_table=self.lookup_table
            )
        get_skipped = lambda individual: score_skipped(
            individual=individual, lookup_table=self.lookup_table)
        return [get_naswot, get_logsynflow, get_skipped]

    def solve(self, max_generations:int=100)->Union[Individual, float]: 
        """
        This function performs Regularized Evolutionary Algorithm (REA) with Training-Free metrics. 
        Details on the whole procedure can be found here: https://arxiv.org/pdf/2207.05135.pdf. 
        
        Args: 
            max_generations (int, optional): TODO - ADD DESCRIPTION. Defaults to 100.
        
        Returns: 
            Union[Individual, float]: Index-0 points to best individual object whereas Index-1 refers to its test 
                                      accuracy.
        """
        
        MAX_GENERATIONS = max_generations
        population, individuals = self.population, self.population.individuals

        for gen in tqdm(range(MAX_GENERATIONS)):
            # perform ageing
            population.age()
            # obtain parents
            parents = self.genetic_operator.obtain_parents(population=individuals)
            # obtain recombinant child
            child = self.perform_recombination(parents=parents)
            # mutate parents
            mutant1, mutant2 = [self.perform_mutation(parent) for parent in parents]
            # add mutants and child to population
            population.add_to_population([child, mutant1, mutant2])
            # preprocess the new population - TODO: Implement a only-if-extremes-change strategy
            self.preprocess_population()
            # remove from population worst (from fitness perspective) individuals
            population.remove_from_population(attribute="fitness", n=2)
            # prune from population oldest individual
            population.remove_from_population(attribute="age", ascending=False)
            # overwrite population
            individuals = population.individuals

        best_individual = max(population.individuals, key=lambda ind: ind.fitness)

        # retrieve test accuracy for this individual
        test_metrics = read_test_metrics(dataset=self.dataset)
        test_accuracy = test_metrics[best_individual.index, 1]

        return (best_individual, test_accuracy)
