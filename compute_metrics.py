"""Pre-computes the metrics in `metrics` for each candidate architecture in the considered search-space."""
from commons import *
from metrics import *
from tqdm import tqdm
from multiprocessing import Pool
from typing import Text, List
import argparse
from scipy.stats import kendalltau, spearmanr
from itertools import product
from commons.utils import init_model
import os

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to be used in computing the considered metrics")
    parser.add_argument("--path-to-save", default="cachedmetrics", type=str, help="Where to save the cached-metrics files")

    return parser.parse_args()

args = parse_args()

this_dataset = args.dataset
cachedmetrics_path = args.path_to_save


def score_all_nets(dataset:str="cifar10", metrics:list=all_metrics, path_to_save:str="cachedmetrics", n_batches:int=3, n_inits:int=3)->None: 
    """Score all networks inside NATS-Bench for the same set of three metrics
    Args: 
        dataset (str, optional): Dataset to be considered in computing the metric. As the architectures are the same across the different datasets
                                 this is not very relevant here, hence the default. Defaults to "cifar10".
        metrics (list, optional): Metrics to be computed. Defaults to `all_metrics`.
        path_to_save (str, optional): Where to save the pre-computed metrics. Defaults to `cachedmetrics`.
        n_batches (int, optional): Number of different batches to test the architecture for. Defaults to 3.
        n_inits (int, optional): Number of different random networks init for the test architecture. Defaults to 3.

    Returns: 
        None. Cachedmetrics is directly saved. 
    """
    nats = NATSInterface(dataset=dataset)
    result = np.zeros(shape=(len(nats), 1+len(metrics)))
    p_bar = tqdm(nats)
    p_bar.set_description(f"Dataset: {dataset}")

    # improve stability with computing the metric over 3 different batches of input data
    batches_list = [
        load_images(dataset=dataset, batch_size=64, with_labels=False) for _ in range(n_batches)
    ]
    for net_idx, net in enumerate(p_bar):
        # improve stability with computing the metric with 3 random initializations
        nets_init = [
            init_model(model=net[0]) for _ in range(n_inits)  # first element of the tuple is a TinyNetwork object.
        ]
        metric_input_net = [(net_init, net[1]) for net_init in nets_init]
        """
        The following equals to:
        result = []
        for metric in metric: 
            metric_avg = []
            for net_tuple in metric_input_net:
                for batch in batches_list:
                    metric_avg.append(
                        metric(net=metric_interface(metric, net_tuple), inputs=batch)
                    )
            result.append(np.mean(metric_avg))
        However, list comprehensions make the following code faster.
        """
        # store the metrics
        # --for debugging purposes only :)
        import time
        start = time.time()
        result[net_idx, :] = [int(net_idx)] + [
            np.mean([
            # computes the average of this metric over 3 batches of images per 3 different initializations
                metric(net=metric_interface(metric, net_tuple), inputs=batch) 
                for batch, net_tuple in product(batches_list, metric_input_net)
                ])
            
            for metric in metrics
            ]
        one_row = time.time() - start
        if False: # change to True for debugging
            print("{:.4g}".format(one_row))
        # --end of debugging :)
    
    np.savetxt(f"{path_to_save}/{dataset}_cachedmetrics.txt", result, header="Arch_Idx, NASWOT, logSynflow, PortionSkipped")
    print(f"{dataset}_cachedmetrics.txt saved at {path_to_save}.")


def performance_all_nets_dataset(dataset:str, training_epochs:int=200, path_to_save:str="cachedmetrics")->None:
    """Retrieves and saves the test accuracy, training time (both per epoch and total) of all networks given a considering dataset.
    
    Args: 
        dataset (str): Dataset to instantiate the API for.
        training_epochs (int, optional): Value of training epochs considered.
        path_to_save (str, optional): Whether to save the cached metrics used.
    
    Raises:
        ValueError: training_epochs is either 12 or 200.
    
    Returns: 
        None.
    """
    if training_epochs not in [12, 200]: 
        raise ValueError(f"Training Epochs accuracy logged only for 12 or 200 epochs! Prompted {training_epochs}")
    
    api = NATSInterface(dataset=dataset)
    results = np.zeros((len(api), 4))  # index column + accuracy, epoch time, total time for training

    pbar = tqdm(range(len(api)))
    pbar.set_description(f"Iterating over all datasets")
    for idx in pbar:
        results[idx, :] = [
            idx,
            # test-accuracy
            api.query_test_performance(architecture_idx=idx, n_epochs=training_epochs)["accuracy"],
            # training per-epoch training time
            api.query_training_performance(architecture_idx=idx, n_epochs=training_epochs)["per-epoch_time"],
            # training all-epochs training time
            api.query_training_performance(architecture_idx=idx, n_epochs=training_epochs)["total_time"]
        ]

    np.savetxt(f"{path_to_save}/{dataset}_TrainTestMetrics.txt", results, header="ArchitectureIdx, Test-Accuracy, TrainingTime(xEpoch), TrainingTime(total)")
    print(f"{dataset}_TrainTestMetrics.txt saved at {path_to_save}.")

def performance_all_nets(datasets:List[Text]=["cifar10", "cifar100", "ImageNet16-120"])->None:
    """Returns a txt file with accuracies and training-stats for all networks in the API.
    
    Args: 
        datasets (List[Text], optional): list of all datasets considered for the usual benchmarks. Can also be reduced to some sub-element.
    """
    with Pool() as pool:
        pool.map(performance_all_nets_dataset, datasets)

def correlation(dataset:str, metric:str, corr_type:str="spearman", read:bool=True, cachedmetric:str="cachedmetrics", verbose:int=None)->float:
    """Computes corr_type correlation between a given metric and the test accuracy on a given dataset.

    Args:
        dataset (str): Dataset to consider for correlation.
        metric (str): Metric to be considered in computing correlation. Must be one of the ones in metrics_names.
        corr_type (str, optional): Correlation type.
        read (bool, optional): Whether or not to read the metric value instead of recomputing them. Defaults to True.
        cachedmetrics (str, optional): The folder where to find the cached-metrics. Defaults to cachedmetrics.
        verbose (int, optional). When >0, prints execution-related information. Defaults to None.
    
    Raises: 
        ValueError: when metric not in metrics_names (available on __init__ of metrics) or corr_type not in ["spearman", "kendall-tau"].

    Returns: 
        float: Correlation between the given metric and test accuracy.
    """
    metric = metric.lower()
    corr_type = corr_type.lower()

    if metric not in metrics_names:  # all metrics implemented so far
        raise ValueError(f"Metric {metric} not among implemented ones: {metrics_names}")
    if corr_type not in ["spearman", "kendall-tau"]:  # types of correlation considered
        raise ValueError(f"{corr_type} correlation not one of ['spearman', 'kendall-tau']")
    
    if not read: 
        raise NotImplementedError("Recomputing metrics from scratch not yet implemented")
    
    file_name = f"{cachedmetric}/{dataset}_cachedmetrics.txt"
    accuracy_file = f"{cachedmetric}/{dataset}_TrainTestMetrics.txt"
    metric_values = np.loadtxt(file_name)[:, 1+metrics_names.index(metric)]  # retrieving column associated with a given metric name
    metric_values[metric_values == -np.inf] = 0
    
    testacc_values = np.loadtxt(accuracy_file)[:, 1]  # accuracy is always at column index 1
    
    if corr_type == "kendall-tau":
        corr = kendalltau(x=metric_values, y=testacc_values).statistic
        if verbose:
            print(f"Kendal-Tau correlation ({metric.upper()}vsTESTACCURACY): "+"{:.4f}".format(corr))
        return corr

    elif corr_type == "spearman":
        corr = spearmanr(a=metric_values, b=testacc_values).statistic
        if verbose:
            print(f"Spearman-R correlation ({metric.upper()}vsTESTACCURACY): "+"{:.4f}".format(corr))
        return corr

def obtain_correlations(
    datasets:List[Text] = ["cifar10", "cifar100", "ImageNet16-120"], 
    metrics:List[Text] = ["naswot", "log-synflow", "portion-skipped-layers"], 
    corr_types:List[Text] = ["spearman", "kendall-tau"]
    )->None:
    """
    Computes correlation values for various combinations of datasets, correlation type and metrics considered. Correlation of metrics is 
    always mesured with respect to test accuracy.
    """
    for dataset, metric, corr_type in tqdm(product(datasets, metrics, corr_types)):
        c = correlation(dataset=dataset, metric=metric, corr_type=corr_type)
        outline = f"On {dataset} {metric} has a {corr_type}-correlation of " + "{:.4g}".format(c)
        print(outline)

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-correlation", action="store_false", help="Stop printint out metrics correlation")
    parser.add_argument("--unify", action="store_true", help="Average all measurements over the different datasets.")
    parser.add_argument("--dataset", default=None, type=str, help="Dataset to which run the metrics on.")

    return parser.parse_args()

args = parse_args()

def check_and_compute(dataset:str="cifar10", cachedmetrics_path:str="cachedmetrics", verbose:int=1):
    """
    This function checks whether or not the cachedmetrics related to an input dataset are present in cachedmetrics_path.
    If not, it computes those.
    """
    if not os.path.exists(f"{cachedmetrics_path}/{dataset}_cachedmetrics.txt"):
        if verbose > 0: 
            print(f"Architectures are not scored over dataset {dataset}. Starting scoring (might take some time...)")
        score_all_nets(dataset=dataset, path_to_save=cachedmetrics_path)

def main():
    cachedmetrics_path = "cachedmetrics"  # change here to store cached metrics somewhere else
    verbosity = 1 # change to > 0 to visualize info as the code runs
    datasets = ["cifar10", "cifar100", "imagenet"]
    """Test whether or not all datasets have been used for scoring. When this is not the case, do so."""
    if args.dataset is None: 
        # score all datasets
        for d in datasets:
            check_and_compute(dataset=d, cachedmetrics_path=cachedmetrics_path, verbose=verbosity)
    # if user provides a single dataset scoring is applied to this only.
    elif args.dataset is not None: 
        check_and_compute(dataset=args.dataset, cachedmetrics_path=cachedmetrics_path, verbose=verbosity)
    else:
        raise ValueError(f"{args.dataset} is not a valid entry!")

    """Unifying the scores over all the datasets. Actually doing it only on users input."""
    unify=args.unify
    if unify:
        avg_metrics = np.mean(
            [np.loadtxt(f"{cachedmetrics_path}/{d}_cachedmetrics.txt", skiprows=1) for d in dataset], 
            axis=0
        )
        output_filename = f"{cachedmetrics_path}/avg_cachedmetrics.txt"
        np.savetxt(
            output_filename, 
            avg_metrics, 
            header="Arch_Idx, NASWOT, logSynflow, PortionSkipped"
        )
        print(f"Unified metrics available at: {output_filename}")

    """Obtain correlations between metrics defined in metrics/__init__.py and test accuracy."""
    stop_correlation=args.stop_correlation
    if not stop_correlation:
        obtain_correlations()

if __name__ == "__main__":
    main()
