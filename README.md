# Fitness Landscape Analysis of Graph Neural Network Architecture Search Spaces  
## The Genetic and Evolutionary Computation Conference (GECCO) - 2021  
### Matheus Nunes, Paulo M. Fraga and Gisele L. Pappa  

#### Overview  

This repository contains code from the original GraphNAS [repository](https://github.com/GraphNAS/GraphNAS) and from the \[Nunes & Pappa, 2020\] [fork](https://github.com/mhnnunes/nas_gnn) . It holds the code for the Evolutionary Algorithm ([here](graphnas/evolution_trainer.py)) and Random Search ([here](graphnas/rs_trainer.py)) strategies, implemented for this paper.  

#### Requirements  

Recent versions of PyTorch, numpy, scipy, dgl, and torch_geometric are required.  

We have provided a utility script that installs the dependencies, considering the usage of CUDA 10.1. If this is not your CUDA version, follow the instructions on the script.  

Example run:  

```{bash}  
./virtualenv_script.sh /opt/cache # use this parameter if you would like to use a different dir. as pip's cache  
```  

After executing this script, you will have an Anaconda powered virtual environment called py37 with the dependencies necessary to run the code in this repository.  

#### Running the code  

We have made available a script for generating the experiment combinations used in the paper. Just run:  

```{bash}  
./generate_experiment_combinations.sh [ea|rs|rl]  
```  

The parameter is the desired optimizer: one of {ea, rl, rs}.  

#### Results  

The results are summarized into a jupyter notebook ([here](1.result_analysis.ipynb)). If you would like to re-execute the notebook, please de-compress the results files `macro_results.tar.xz` and `micro_results.tar.xz`.  

```{bash}  
tar -xvf macro_results.tar.xz  
```  

#### Acknowledgements  
This repo is modified based on [DGL](https://github.com/dmlc/dgl), [PYG](https://github.com/rusty1s/pytorch_geometric) and [GraphNAS](https://github.com/GraphNAS/GraphNAS).  
