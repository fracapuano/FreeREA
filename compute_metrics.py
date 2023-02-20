"""Pre-computes the metrics in `metrics` for each candidate architecture in the considered search-space."""
from commons import *
from metrics import *
from tqdm import tqdm

import time

def score_all_nets(dataset:str="cifar10", metrics:list=all_metrics): 
    """Score all networks inside NATS-Bench for the same set of three metrics"""
    nats = NATSInterface(dataset=dataset)
    result = np.zeros(shape=(len(nats), len(metrics)))
    # taking a random batch of images
    images = Dataset(name=dataset).random_examples(with_labels=False)

    for net_idx, net in enumerate(tqdm(nats)):
        start = time.time()
        result[net_idx, :] = [
            metric(net=metric_interface(metric, net), inputs=images) for metric in metrics
            ]
        compute_time = time.time() - start
        print(f"Time to score one candidate: {compute_time}")
        break

def main():
    score_all_nets()

if __name__ == "__main__":
    main()
