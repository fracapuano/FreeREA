from search import FreeREA
import argparse

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to be considered. One in ['cifar10', 'cifar100', 'ImageNet16-120'].s")
    parser.add_argument("--n-generations", default=100, type=int, help="Number of generations to let the genetic algorithm run.")
    parser.add_argument("--lookup", action="store_true", help="When provided, uses lookup table instead of computing metrics on the fly.")
    
    parser.add_argument("--default", action="store_true", help="Default configuration. Ignores evvery other parameter when specified")
    return parser.parse_args()

args = parse_args()

dataset=args.dataset
n_generations=args.n_generations
use_lookup=args.lookup

if args.default: 
    dataset="cifar10"
    n_generations=100
    use_lookup=True

def main():
    """Invokes the solve method and prints out information on best individual found"""
    algorithm = FreeREA(dataset=dataset, lookup=use_lookup)

    best_individual, test_accuracy = algorithm.solve(
        max_generations=n_generations
    )
    print(f"Best Individual: {best_individual.genotype}, with the following scores:")
    print(f'    "naswot_score = {best_individual.naswot_score}')
    print(f'    "logsynflow_score = {best_individual.logsynflow_score}')
    print(f'    "skip_score = {best_individual.skip_score}')
    print(f'Its test accuracy is: {test_accuracy}')

if __name__=="__main__":
    main()