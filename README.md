*Edit: The authors official implementation is now available on GitHub at [github.com/NiccoloCavagnero/FreeREA](https://github.com/NiccoloCavagnero/FreeREA).*

> **Disclaimer**
This repo offers a tour of an implementation of [FreeREA: Training-Free Evolution-based Architecture Search](https://arxiv.org/pdf/2207.05135.pdf), by Cavagnero et al, 2022.
This work is not associated in any way or form with the authors and only aims at reproducing the findings presented by the authors in the cited paper. 
The main author has been informed of the publicity of this re-implementation. 


# Set-up

After having downloaded the repo, the first step to use this code is installing dependencies.

To set up dependancies, set up a virtual environment clonig the `env.yml` file.

```bash
conda env create -f env.yml
```

Once dependencies are successfully installed, please go ahead and download **in the main folder** (that is, `freeREA`) the `archive` folder, containing the actual NATS-Bench networks. Archive also contains data that are otherwise re-downloaded. `archive` is already in the `.gitignore` file of this repo. 

To download `archive` run the following:
```bash
wget "https://www.dropbox.com/sh/ceeo70u1buow681/AADxyCvBAnE6mMjp7JOoo0LVa/NATS-tss-v1_0-3ffb9-simple.tar"
tar -xf "NATS-tss-v1_0-3ffb9-simple.tar"
mv NATS-tss-v1_0-3ffb9-simple.tar archive/NATS-tss-v1_0-3ffb9-simple.tar
```
This solution only downloads the NATS search space and triggers the download of the CIFAR10 dataset.

Alternatively, one could download `archive` from [here](https://drive.google.com/file/d/1LMpDiS1hmCLsC4Y86bhF41NzqAx5kS8c/view) and then unzip the folder. 

Please consider that downloading the search space only is more than sufficient as fully trained models are not needed, since the benchmark conveniently stores the model performance metrics. More than that, the trained models are very space intensive.

# Examples
To see an example of how this repository can be used, please refer to [this notebook](https://github.com/fracapuano/freeREA/blob/main/FreeREA.ipynb).
Moreover, a more generic example of usage can be found [here](https://github.com/gsuriano/Project8_Group5/blob/main/Group5_Step3.ipynb).
