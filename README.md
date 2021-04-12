# Fitness Landscape Analysis of Graph Neural Network Architecture Search Spaces  
## The Genetic and Evolutionary Computation Conference (GECCO) - 2021  
### Matheus Nunes, Paulo M. Fraga ([github](https://github.com/paulohmf)) and Gisele L. Pappa  

#### Overview  

This repository contains code from the original GraphNAS [repository](https://github.com/GraphNAS/GraphNAS) and from the \[Nunes & Pappa, 2020\] [fork](https://github.com/mhnnunes/nas_gnn).  

In this paper, we extend the work of [Nunes & Pappa, 2020](https://link.springer.com/chapter/10.1007/978-3-030-61377-8_21) by:  
1. Expanding the search space for each dataset (as done [here](src/expand_cora_macro.py))  
2. Encoding the architectures in two different ways ([here](src/architecture_analysis/generate_encode_macro_full.py))  
3. Calculating FLA metrics for both representations ([here](src/architecture_analysis/1.graph_result_analysis.ipynb) and [here](src/architecture_analysis/tsne_analysis.py))  

The results can be found in this [notebook](src/architecture_analysis/1.graph_result_analysis.ipynb).  

#### Requirements  

Recent versions of `PyTorch`, `numpy`, `pandas`, `scipy`, `dgl`, and `torch_geometric` are required.  

We have provided a utility script that installs the dependencies, considering the usage of CUDA 10.1. If this is not your CUDA version, follow the instructions on the script.  

Example run:  

```{bash}  
./virtualenv_script.sh  
```  

After executing this script, you will have an Anaconda powered virtual environment called py37 with the dependencies necessary to run the code in this repository.  

#### Running the code  

The execution of the code for this paper is divided into two parts: a server that evaluates the architectures, and the client that requests the evaluations. In order to run the server, just `cd` into the `src` directory and run:  

```{bash}  
python -u -m server_client.evaluator_server -p 12345    
```  

This command initiates a server on port `12345`.

In order to execute the script that runs the architecture evaluations in this work, just run the codes in the first item of the list in section [Overview](#overview).  


#### Acknowledgements  
This repo is modified based on [DGL](https://github.com/dmlc/dgl), [PYG](https://github.com/rusty1s/pytorch_geometric) and [GraphNAS](https://github.com/GraphNAS/GraphNAS).  
