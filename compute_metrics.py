"""Pre-computes the metrics in `metrics` for each candidate architecture in the considered search-space."""
from commons import *
from metrics import *
from tqdm import tqdm
from multiprocessing import Pool

import argparse


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

def score_all_nets(dataset:str="cifar10", metrics:list=all_metrics, path_to_save:str="cachedmetrics"): 
    """Score all networks inside NATS-Bench for the same set of three metrics"""
    nats = NATSInterface(dataset=dataset)
    result = np.zeros(shape=(len(nats), 1+len(metrics)))
    # taking a random batch of images
    images = Dataset(name=dataset, batchsize=64).random_examples(with_labels=False)
    p_bar = tqdm(nats)
    p_bar.set_description(f"Dataset: {dataset}")

    for net_idx, net in enumerate(p_bar):
        # storing the metrics
        result[net_idx, :] = [net_idx] + [
            metric(net=metric_interface(metric, net), inputs=images) for metric in metrics
            ]
        
    np.savetxt(f"{path_to_save}/{dataset}_cachedmetrics.txt", result, header="Arch_Idx, NASWOT, logSynflow, PortionSkipped")

def main():
    score_all_nets(dataset=this_dataset, path_to_save=cachedmetrics_path)

if __name__ == "__main__":
    main()
