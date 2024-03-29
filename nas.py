from search import GeneticSearch, FreeREA_dict
from commons import seed_all, NATSInterface, genotype_to_architecture, plot_mean_with_ci_fillbetween
import numpy as np
import argparse
from tqdm import tqdm
import os
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt

"""TODO: run parallele non funzionano. Ottengo "troppi file aperti" come messaggio di errore."""

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
    parser.add_argument("--goparallel", action="store_true", help="When provided, triggers different runs to be parallel rather than sequentially.")
    
    parser.add_argument("--default", action="store_true", help="Default configuration. Ignores evvery other parameter when specified")
    return parser.parse_args()

args = parse_args()

dataset=args.dataset
n_generations=args.n_generations
n_runs=args.n_runs
use_lookup=args.lookup
store_traj=args.store_run
savefig=args.savefig
goparallel=args.goparallel

if args.default: 
    dataset="cifar10"
    n_generations=50
    n_runs=30
    use_lookup=True

def init_and_launch():
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
    show = False  # whether or not to show the figure with final results
    # search space interface
    interface = NATSInterface(dataset=dataset)
    
    if goparallel:  # parallelize the execution of the various test runs
        n_cpus = min(int(n_runs/2.5), int(os.cpu_count() * .50))  # use 50% available cores
        print(f"Using {n_cpus} cores!")
        # shared memory object to which all subprocesses can access
        result_list = Manager().list()
        with Pool(processes=n_cpus) as pool:
            p = pool.imap_unordered(launch_and_log, [result_list for _ in range(n_runs)]) # this list comprehension points at the same object in memory
            for _ in tqdm(p, total=n_runs):
                pass # simply updates the progress bar without doing anything else
    else: 
        result_list = []
        for _ in tqdm(range(n_runs)): 
            launch_and_log(result_list=result_list)
        
    # interfacing the output
    if store_traj: 
        accuracies = [r[0] for r in result_list]
        nets = [r[1][-1].genotype for r in result_list]
    else: 
        accuracies = result_list
    
    script_dir = os.path.dirname(__file__)
    logs_dir = os.path.join(script_dir, 'logs/')
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    with open(logs_dir + f"{n_generations}gens_{dataset}_log.txt", "w") as f:
        f.write("Terminal timestep nets reached are: \n")
        for idx, net in enumerate(nets):
            f.write(f"Run-{idx}: " + genotype_to_architecture(net) + "\n")
    # retrieving average and std for the accuracy
    avg_test_accuracy, std_test_accuracy = np.mean(accuracies), np.std(accuracies)

    print('On {} (using {} gens) the found networks have an average test accuracy of: {:.5g} ± {:.5g}'.format(
        dataset, n_generations, avg_test_accuracy, std_test_accuracy)
        )
    
    if store_traj:
        trajectories = np.array(
            [np.fromiter(map(lambda ind: genotype_to_accuracy(interface=interface, genotype=ind.genotype), r[1]), dtype=float) for r in result_list]
            ).reshape((n_runs, -1))
        # plot average evolution alongside 95% CI over the mean
        fig, ax = plot_mean_with_ci_fillbetween(trajectories)
        if show: 
            plt.show()
        # choose whether or not to save the figure
        if savefig:
            results_dir = os.path.join(script_dir, 'images/')
            exp_figname = f"AvgEvolution_{dataset}_{n_runs}runs_{n_generations}gens.svg"
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            
            fig.savefig(results_dir + exp_figname)
        
if __name__=="__main__":
    main()
