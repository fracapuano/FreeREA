from commons import *
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",
                        default=10,
                        type=int,
                        help="Number of random samples to draw from NATS bench")

    return parser.parse_args()


args = parse_args()


def main():
    NATS_PATH = str(get_project_root()) + "/archive/NATS-tss-v1_0-3ffb9-simple/"
    dataset = "CIFAR10"
    nats = NATSBench(path=NATS_PATH, dataset=dataset)
    # create nats object with all architectures instantiated - NAS-Bench 201 corresponds with "topology" search space
    nats_api = create(file_path_or_dict=NATS_PATH, search_space="topology", fast_mode=True)
    for _ in range(args.n_samples):
        # sampling a random architecture from the search space considered on cifar10
        random_idx = nats_api.random()
        random_architecture = nats_api.get_net_config(index=random_idx, dataset="cifar10")["arch_str"]
        random_idx_interface = nats.get_index(architecture=random_architecture)
        print(random_idx, random_idx_interface)

        # turning the architecture into a network
        random_network = Exemplar(space=nats, idx=random_idx, genotype=random_architecture)
        random_network = nats.get_network(architecture=random_network)

        # print(f"In NATS Bench, Network {random_idx}'s architecture is:")
        # print(random_network)
        print(50*"*")

if __name__ == "__main__":
    main()
