=======
FlowGNN
=======

   Official Python code for the paper "Flow Graph Neural Networks" submitted to ICLR 2024:
   https://openreview.net/forum?id=iKI7wT6fCP

This repository is divided into two distinct sections: flowgnn_opamps and flowgnn_powergraph. 

The flowgnn_opamps folder contains the code for the experiments on the CktBench-101 dataset from Dong et al. (2023) (Paper: https://arxiv.org/abs/2308.16406, Code: https://github.com/zehao-dong/CktGNN). 

The flowgnn_powergraph folder contains the code for the experiments on the PowerGraph dataset. Compared to the original PowerGraph-Graph code from Varbella et al. (2024) (Paper: https://arxiv.org/abs/2402.02827, Code: https://github.com/PowerGraph-Datasets/PowerGraph-Graph), we added some more models: GATv2, FlowGAT, FlowGATv2, FlowTransformer (FlowGT).

.. image:: ./flowgnn.png
    :height: 450px
