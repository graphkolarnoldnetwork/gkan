<img width="969" alt="GKAN_architectures" src="https://github.com/user-attachments/assets/03649a50-4a2e-4c32-943e-c33db7ade918">

# Graph Kolmogorov-Arnold Networks (GKANs)

This is the github repo for the paper "GKAN: Graph Kolmogorov-Arnold Networks". 

Graph Kolmogorov-Arnold Networks (GKANs) are an innovative extension of the Kolmogorov-Arnold Networks (KANs) for graph-structured data. By leveraging the unique attributes of KANs, particularly the use of learnable univariate functions instead of fixed linear weights, GKANs offer a powerful approach for graph-based learning tasks. Unlike traditional Graph Convolutional Networks (GCNs) that depend on a fixed convolutional architecture, GKANs employ learnable spline-based functions between layers, revolutionizing information processing across graph structures.

We propose two distinct architectures to integrate KAN layers into GKANs:

Architecture 1: where learnable functions are applied to input features after aggregation.
Architecture 2: where learnable functions are applied to input features before aggregation.

Empirical evaluations on a semi-supervised graph learning task using the Cora dataset demonstrate that GKANs outperform traditional GCN models. For instance, with 100 features, GCN achieves an accuracy of 53.5%, whereas a GKAN with a comparable number of parameters attains an accuracy of 61.76%. With 200 features, GCN achieves 61.24%, while GKAN achieves 67.66%. Additionally, we investigate the impact of various parameters such as the number of hidden nodes, grid size, and the polynomial degree of the spline on GKAN performance.

GKANs show promising improvements in accuracy and flexibility, providing a new direction for graph representation learning. 

We also modified [KAN implementation] which could not support the cases with large number of features, such as Cora (with 1,433 features) and Citeseer datasets (with 3,703 features). In this implementation, we applied GKAN on the following three graph-structure datases:

## Dataset Information

| Dataset   | Type              | Nodes | Edges | Classes | Features |
|-----------|-------------------|-------|-------|---------|----------|
| Citeseer  | Citation network  | 3,327 | 4,732 | 6       | 3,703    |
| Cora      | Citation network  | 2,708 | 5,429 | 7       | 1,433    |
| Pubmed    | Citation network  | 19,717| 44,338| 3       | 500      |


<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

## Accuracy of GKAN compared to GCN
Not only does GKANs provides higher accuracy compared to GCNs for the case of considering same number of parapemnters (to have a fair comparsion), but also they require considerably few number of epochs to lead to the same performance as GCNs.



**GKANs outperforms GCNs**
Here are the results of applying GKAN models as well as GCN on Cora, Citeseer, and Pubmed datasets:

## On Cora dataset with the first 100 features of Cora Dataset

| Architecture                         | #Parameters | Test  |
|--------------------------------------|-------------|-------|
| GCN (h_GCN = 205)                    | 22,147      | 53.50 |
| GKAN (Archit. 1) (k=1, g=10, h=16)   | 22,279      | 59.32 |
| GKAN (Archit. 2) (k=1, g=10, h=16)   | 22,279      | 61.48 |
| GKAN (Archit. 1) (k=2, g=9, h=16)    | 22,279      | 56.76 |
| GKAN (Archit. 2) (k=2, g=9, h=16)    | 22,279      | **61.76** |


## On Cora dataset with the first 200 features of Cora Dataset

| Architecture                         | #Parameters | Test  |
|--------------------------------------|-------------|-------|
| GCN (h_GCN = 104)                    | 21,639      | 61.24 |
| GKAN (Archit. 1) (k=2, g=2, h=17)    | 21,138      | 63.58 |
| GKAN (Archit. 2) (k=2, g=2, h=17)    | 21,138      | 64.10 |
| GKAN (Archit. 1) (k=1, g=2, h=20)    | 20,727      | 67.44 |
| GKAN (Archit. 2) (k=1, g=2, h=20)    | 20,727      | **67.66** |

## On Citeseer dataset with the first 100 features

| Architecture                         | #Parameters | Test  |
|--------------------------------------|-------------|-------|
| GCN (h_GCN = 55)                     | 5,891       | 36.2  |
| GKAN (Archit. 1) (k=2, g=1, h=10)    | 5,316       | 41.6  |
| GKAN (Archit. 2) (k=2, g=1, h=10)    | 5,316       | 40.5  |
| GKAN (Archit. 1) (k=1, g=2, h=10)    | 5,316       | 43.1  |
| GKAN (Archit. 2) (k=1, g=2, h=10)    | 5,316       | **43.3** |

## On Pubmed dataset with the first 100 features

| Architecture                         | #Parameters | Test  |
|--------------------------------------|-------------|-------|
| GCN (h_GCN = 50)                     | 5,203       | 68.8  |
| GKAN (Archit. 1) (k=2, g=1, h=10)    | 5,163       | 70.1  |
| GKAN (Archit. 2) (k=2, g=1, h=10)    | 5,163       | 72.1  |
| GKAN (Archit. 1) (k=1, g=2, h=10)    | 5,163       | **73.5** |
| GKAN (Archit. 2) (k=1, g=2, h=10)    | 5,163       | 71.9  |


To get

## Installation
Pygkan can be installed via GitHub. 

**Pre-requisites:**

```
Python 3.9.7 or higher
pip
```

**Installation via github**

```
python -m venv pykan-env
source pykan-env/bin/activate  # On Windows use `pykan-env\Scripts\activate`
pip install git+https://github.com/KindXiaoming/pykan.git
```

**Installation via PyPI:**
```
python -m venv pygkan-env
source pygkan-env/bin/activate  # On Windows use `pygkan-env\Scripts\activate`
```
Requirements

```python
# python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```



## Computation requirements

Codes are runnable on a single CPU in case of using medium scale number of input features. In case of using large number of input features for your task or using large dataset (large number of nodes), you may need to run the code on GPU. We advice to use batches to speed up the process. 



**Runnin the code**

Get started with 'gkan.py' by following commands:

```ssh
cd pygkan
python gkan.py
```

Files 'inlfuence_g_gkan.py', 'inlfuence_k_gkan.py', and 'inlfuence_h_gkan.py' are the codes corresponds to the effect of parameters g, k, and h, respectively.

## Advice on hyperparameter tuning and 

* We consider a simple setup (small GKAN shape, small grid size g, and small degree of polynomial).

* Once an acceptable performance is achieved, you could then try refining your GKAN.

* If you care about accuracy, try grid extention, increasing hidden layer size as well as polynomial degree. 


Please cite GKAN paper if you use this code in your own work.

## Contact
If you have any questions, please contact graphkolarnoldnetwork@gmail.com

## Author's note
I am deeply grateful to everyone who has shown interest in GKANs. I modified the KAN implementation to support datasets with a number of features in order of thousands, as the original KAN implementation could not handle such cases. I welcome any criticism regarding the efficiency and reusability of the code and apologize for any inconvenience. I simply hope you enjoy using the code for your own datasets and models.




