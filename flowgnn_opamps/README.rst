.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==============
FlowGNN on Ckt-Bench101
==============


    Code for the experiments with the FlowGNN models on the operational amplifiers from the Ckt-Bench101 dataset.


==============
Installation
==============

**Requirements**

- CPU or NVIDIA GPU
- Linux or Windows
- Python 3.9
- PyTorch 2.4.0
- PyTorch Geometric 2.6.0

.. code-block:: python

   conda create -n flowgnn-env python=3.9
   conda activate flowgnn-env
   pip3 install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu118
   pip3 install torch_geometric==2.6.0
   pip3 install torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
   pip3 install -r requirements_{os}.txt # Install remaining packages, e.g. igraph. Linux: os=linux, Windows: os=win.
   pip3 install . # In flowgnn_opamps folder


==============
Data
==============

The Ckt-Bench101 dataset is available at https://github.com/zehao-dong/CktGNN/tree/main/OCB/CktBench101.

Download the following files and place them inside the "data" folder:

- ckt_bench_101.pkl (Graphs)
- perform101.csv (Labels)

The graphs contained in "ckt_bench_101.pkl" are in igraph-format. To convert them into torch-geometric-2.6 format,
use the notebook "data_conversion.ipynb". It will create a file called "ckt_bench_101_tg26.pkl", which can be used
to train the models.

==============
Training
==============

To train any model, use the script "supervised_experiment.py". Specify your settings in the script ("Experimental 
settings" and "Training hyperparameters") and run it via

.. code-block:: python

    python supervised_experiment.py

Feel free to use the notebook "experiment_analysis.ipynb" to plot training curves and leaderboards.



.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
