from search import GeneticSearch, FreeREA_dict
from commons import seed_all
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
import os
from tqdm import tqdm

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to be considered. One in ['cifar10', 'cifar100', 'ImageNet16-120'].s")
    parser.add_argument("--n-generations", default=50, type=int, help="Number of generations to let the genetic algorithm run.")
    parser.add_argument("--n-runs", default=30, type=int, help="Number of runs used to ")
    parser.add_argument("--lookup", action="store_false", help="When provided, uses lookup table instead of computing metrics on the fly.")
    
    parser.add_argument("--default", action="store_true", help="Default configuration. Ignores evvery other parameter when specified")
    return parser.parse_args()

args = parse_args()

dataset=args.dataset
n_generations=args.n_generations
n_runs=args.n_runs
use_lookup=args.lookup

if args.default: 
    dataset="cifar10"
    n_generations=50
    n_runs=30
    use_lookup=True

def init_and_launch()->float: 
    """
    This function inits and launches FreeREA.
    It returns the test accuracy.
    """
    # initializes the algorithm
    algorithm = GeneticSearch(
        dataset=dataset, 
        lookup=use_lookup, 
        genetics_dict=FreeREA_dict
    )
    # obtains test accuracy
    _, test_accuracy = algorithm.solve(
        max_generations=n_generations
    )
    return test_accuracy

def main():
    """Invokes the solve method and prints out information on best individual found"""
    seed_all(seed=0)  # FreeREA's seed
    num_cpus = int(0.75 * os.cpu_count()) # use 75% of available cpus
    
    accuracies = [None for _ in range(n_runs)]
    for run_idx in tqdm(range(n_runs), desc="Top test accuracy: {:.4g}".format(test_accuracy)):
        # initialize FreeREA algorithm and launch an experiment
        test_accuracy = init_and_launch()
        # store the accuracy found
        accuracies[run_idx] = test_accuracy

    # retrieving average and std for the accuracy
    avg_test_accuracy, std_test_accuracy = np.mean(accuracies), np.std(accuracies)

    print('On {} the found networks has an average test accuracy of: {:.5g} ± {:.5g}'.format(dataset, avg_test_accuracy, std_test_accuracy))

if __name__=="__main__":
    main()