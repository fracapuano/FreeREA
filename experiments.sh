#!/bin/bash

# Define the arguments that should be looped over
generations=(50 100 150)
datasets=("cifar10" "cifar100" "imagenet")

# Loop over the arguments and run nas.py for each combination
for gen in "${generations[@]}"; do
  for dataset in "${datasets[@]}"; do
    # Define the command to execute
    echo "Executing nas.py with $gen gens on $dataset"
    cmd="python nas.py --n-generations $gen --dataset $dataset --savefig"
    # Execute the command in the background
    $cmd &
  done
done

# Wait for all processes to finish
wait