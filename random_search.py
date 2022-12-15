from nats_bench import create
from utils.utils import *

# path where the pickle files of all architectures in the topological search space are
NATS_PATH = str(get_project_root()) + "/archive/" + "NATS-tss-v1_0-3ffb9-simple"
# create nats object with all architectures instantiated - NAS-Bench 201 corresponds with "topology" search space
nats = create(NATS_PATH, "topology", True)

print(nats.random())