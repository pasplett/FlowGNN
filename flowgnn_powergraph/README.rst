==============
FlowGNN on PowerGraph
==============


    Code for the experiments with the FlowGNN models on the PowerGraph dataset.


==============
Credits
==============

This project is a fork of **PowerGraph-Datasets/PowerGraph-Graph (https://github.com/PowerGraph-Datasets/PowerGraph-Graph)** by **avarbella**,
licensed under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
Modifications copyright © 2025 pasplett.

Changes to the original code also include code from the following repositories:

- https://github.com/phoeenniixx/pytorch_geometric/tree/exphormer (MIT license)

- https://github.com/BorgwardtLab/SAT/tree/main (BSD 3-Clause License)

Original license files can be found in the corresponding folders.

==============
Changes Made
==============

Compared to the original code, we added the following models to gnn.model:

- GraphSAGE
- GATv2
- FlowGAT
- FlowGATv2
- FlowTransformer
- GraphGPS
- SAT
- Exphormer

Further changes:

- Additional evaluation metrics in train_gnn.py

==============
Installation
==============

- CPU or NVIDIA GPU, Linux, Python 3.9
- PyTorch >= 2.4.0, other packages

Install additional packages:

.. code-block:: python

   pip install -r requirements.txt


==============
Data
==============

The PowerGraph dataset is available at https://figshare.com/articles/dataset/PowerGraph/22820534.

For more details, see https://github.com/PowerGraph-Datasets/PowerGraph-Graph.

==============
Training
==============

To train different GNN architectures  (GCN, GAT, FlowGAT, ...) on the PowerGraph dataset, run:

.. code-block:: python

    python code/train_gnn.py

For more details, see https://github.com/PowerGraph-Datasets/PowerGraph-Graph.

==============
License
==============

This work is licensed under a CC BY 4.0 license.
