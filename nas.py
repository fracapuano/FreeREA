from search import GeneticSearch, FreeREA_dict
from commons import seed_all, NATSInterface, genotype_to_architecture, plot_mean_with_ci_fillbetween
import numpy as np
import argparse
from tqdm import tqdm
import os
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt

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
    parser.add_argument("--store-run", action="store_false", help="When provided, avoids creation of a file in which to store statistics about each run.")
    parser.add_argument("--savefig", action="store_true", help="When provided, triggers saving the fig related to the avg test accuracy obtained during the opt process.")
    
    parser.add_argument("--default", action="store_true", help="Default configuration. Ignores evvery other parameter when specified")
    return parser.parse_args()

args = parse_args()

dataset=args.dataset
n_generations=args.n_generations
n_runs=args.n_runs
use_lookup=args.lookup
store_traj=args.store_run
savefig=args.savefig

if args.default: 
    dataset="cifar10"
    n_generations=50
    n_runs=30
    use_lookup=True

def init_and_launch()->None: 
    """
    This function inits and launches FreeREA.
    It returns the test accuracy.
    """
    # initializes the algorithm
    algorithm = GeneticSearch(
        dataset=dataset,
        lookup=use_lookup,
        genetics_dict=FreeREA_dict,
        #fitness_weights=np.array([0.5, 0.5, 0])  # should always get conv3x3
    )
    # obtains test accuracy
    result = algorithm.solve(
        max_generations=n_generations, 
        return_trajectory=store_traj
    )
    if store_traj:
        _, test_accuracy, trajectory = result   
        return test_accuracy, trajectory     
    else:
        *_, test_accuracy = result
        return test_accuracy

# create function for appending exp results to list
def log_result(target_list:list, result:float):
    """Function handle to append stuff to list shared among processes."""
    target_list.append(
        result
    )

def launch_and_log(result_list:list)->None: 
    """
    Launches the optimization process and stores final test accuracy.
    Dummy arg used to launch this function parallely with others.
    """
    # shared-memory list to which to append the results
    exp_results = result_list
    # result of a single experiment
    exp_result = init_and_launch()
    # logs the experimental result obtained to a shared-memory list
    log_result(target_list=exp_results, result=exp_result)

def genotype_to_accuracy(interface:object, genotype:list)->float: 
    """Returns the fitness of a given architecture, identified through its genotype."""
    # genotype -> arch string -> arch index -> accuracy score (check interface object for more details)
    return interface.query_test_performance(
        interface.query_index_by_architecture(
            genotype_to_architecture(genotype=genotype)
            )
        )["accuracy"]

def main():
    """Invokes the solve method and prints out information on best individual found"""
    seed_all(seed=0)  # FreeREA's seed
    # search space interface
    interface = NATSInterface(dataset=dataset)
    result_list = Manager().list()
    n_cpus = min(n_runs, int(os.cpu_count() * .75))  # use 75% of available cores

    with Pool(processes=n_cpus) as pool:
        p = pool.imap(launch_and_log, [result_list for _ in range(n_runs)]) # this list comprehension points at the same object in memory
        for _ in tqdm(p, total=n_runs):
           pass
    
    # interfacing the output
    if store_traj: 
        accuracies = [r[0] for r in result_list]
    else: 
        accuracies = result_list
    # retrieving average and std for the accuracy
    avg_test_accuracy, std_test_accuracy = np.mean(accuracies), np.std(accuracies)

    print('On {} the found networks has an average test accuracy of: {:.5g} ± {:.5g}'.format(
        dataset, avg_test_accuracy, std_test_accuracy)
        )
    
    if store_traj:
        trajectories = np.array(
            [np.fromiter(map(lambda ind: genotype_to_accuracy(interface=interface, genotype=ind.genotype), r[1]), dtype=float) for r in result_list]
            ).reshape((n_runs, -1))
        # plot average evolution alongside 95% CI over the mean
        fig, ax = plot_mean_with_ci_fillbetween(trajectories)
        plt.show()
        # choose whether or not to save the figure
        if savefig:
            fig.savefig(f"AvgEvolution_{dataset}_{n_runs}.svg")
        
if __name__=="__main__":
    main()