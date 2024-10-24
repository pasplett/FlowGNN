==============
FlowGNN on PowerGraph
==============


    Code for the experiments with the FlowGNN models on the PowerGraph dataset.


This repository is based on the original PowerGraph repository:

https://github.com/PowerGraph-Datasets/PowerGraph-Graph

Compared to the original code, we added the following models to gnn.model:

- GATv2
- FlowGAT
- FlowGATv2
- FlowTransformer

==============
Installation
==============

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

Load every additional packages:

```
pip install -r requirements.txt
```


==============
Data
==============

The PowerGraph dataset is available at https://figshare.com/articles/dataset/PowerGraph/22820534.

For more details, see https://github.com/PowerGraph-Datasets/PowerGraph-Graph.

==============
Training
==============

To test the datasets with different GNN architectures: GCN, GINe, GAT and Transformer, run,

.. code-block:: python

    python code/train_gnn.py

For more details, see https://github.com/PowerGraph-Datasets/PowerGraph-Graph.
