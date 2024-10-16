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
- Linux / Windows
- Python 3.12
- PyTorch 2.4.0
- PyTorch Geometric 2.6.0

.. code-block:: python

   pip install -r requirements.txt
   pip install .


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


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
