import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected


class CktDataset(Dataset):

    def __init__(
        self, graph_file, target_file, target="gain", train=True, val=False, 
        n_val=1000, device=torch.device("cpu"), dtype=torch.float64,
        val_idx=None
    ):
        """PyTorch dataset class for electronic circuits and their 
        corresponding, simulated properties.

        Parameters
        ----------
        graph_file : str
            Path to .pkl-file containing the graph data.
        target_file : str
            Path to .csv-file containing the target values for each graph.
        target : str
            Name of the target variable, by default "gain"
        train : bool, optional
            Whether to use training or test set, by default True
        val : bool, optional
            Whether to use training or validation set, by default False
        n_val : int, optional
            Size of the validation dataset, by default 1000.
        """

        with open(graph_file, "rb") as f:
            graphs = pickle.load(f)

        self.graphs = graphs
        target_df = pd.read_csv(target_file, index_col=0)

        if train:
            self.graphs = self.graphs[0]
            target_df = target_df[:len(self.graphs)]
            if val:
                if not(val_idx is None):
                    self.graphs = [g for i, g in enumerate(self.graphs) if i in val_idx]
                    target_df = target_df.loc[val_idx]
                else:
                    self.graphs = self.graphs[-n_val:]
                    target_df = target_df[-n_val:]
            else:
                if not(val_idx is None):
                    train_idx = [i for i in range(len(self.graphs)) if not i in val_idx]
                    self.graphs = [g for i, g in enumerate(self.graphs) if i in train_idx]
                    target_df = target_df.loc[train_idx]
                else:
                    self.graphs = self.graphs[:-n_val]
                    target_df = target_df[:-n_val]
        else:
            self.graphs = self.graphs[1]
            target_df = target_df[-len(self.graphs):]
            target_df = target_df.reset_index()

        self.targets = torch.Tensor(target_df[target].values)

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]
    

class CktDataset_DAGFormer(Dataset):

    def __init__(
        self, graph_file, target_file, target="gain", train=True, val=False, 
        n_val=1000, device=torch.device("cpu"), dtype=torch.float64,
        val_idx=None, use_mpnn=False
    ):
        """PyTorch dataset class for electronic circuits and their 
        corresponding, simulated properties.

        Parameters
        ----------
        graph_file : str
            Path to .pkl-file containing the graph data.
        target_file : str
            Path to .csv-file containing the target values for each graph.
        target : str
            Name of the target variable, by default "gain"
        train : bool, optional
            Whether to use training or test set, by default True
        val : bool, optional
            Whether to use training or validation set, by default False
        n_val : int, optional
            Size of the validation dataset, by default 1000.
        """

        with open(graph_file, "rb") as f:
            graphs = pickle.load(f)

        self.graphs = graphs
        target_df = pd.read_csv(target_file, index_col=0)

        if train:
            self.graphs = self.graphs[0]
            target_df = target_df[:len(self.graphs)]
            if val:
                if not(val_idx is None):
                    self.graphs = [g for i, g in enumerate(self.graphs) if i in val_idx]
                    target_df = target_df.loc[val_idx]
                else:
                    self.graphs = self.graphs[-n_val:]
                    target_df = target_df[-n_val:]
            else:
                if not(val_idx is None):
                    train_idx = [i for i in range(len(self.graphs)) if not i in val_idx]
                    self.graphs = [g for i, g in enumerate(self.graphs) if i in train_idx]
                    target_df = target_df.loc[train_idx]
                else:
                    self.graphs = self.graphs[:-n_val]
                    target_df = target_df[:-n_val]
        else:
            self.graphs = self.graphs[1]
            target_df = target_df[-len(self.graphs):]
            target_df = target_df.reset_index()

        self.targets = torch.Tensor(target_df[target].values)

        self.use_mpnn = use_mpnn

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        data, y = self.graphs[idx], self.targets[idx]

        edge_index_dag = data.edge_index
        if self.use_mpnn:
            # using mpnn to implement DAG attention
            data.dag_rr_edge_index = to_undirected(edge_index_dag)
        else:
            # using mask to implement DAG attention
            num_nodes = data.num_nodes
            if(num_nodes<=1000):
                max_num_nodes = 1000
                mask_rc = torch.tensor([]).new_zeros(max_num_nodes, max_num_nodes).bool()
                for index1 in range(num_nodes):
                    ne_idx = edge_index_dag[0] == index1
                    le_idx = ne_idx.nonzero(as_tuple=True)[0]
                    lp_edge_index = edge_index_dag[1, le_idx]
                    ne_idx_inverse = edge_index_dag[1] == index1
                    le_idx_inverse = ne_idx_inverse.nonzero(as_tuple=True)[0]
                    lp_edge_index_inverse = edge_index_dag[0, le_idx_inverse]
                    mask_r=torch.tensor([]).new_zeros(max_num_nodes).bool()
                    mask_r[lp_edge_index] = True
                    mask_r[lp_edge_index_inverse] = True
                    # mask_r[index1] = True
                    mask_rc[index1] = ~ mask_r
            data.mask_rc = mask_rc
            
        return data, y
    

class CktDataset_igraph(CktDataset):
    def __init__(self, *args, **kwargs):
        """Special dataset class for igraph datasets. 

        Parameters
        ----------
        subgraph_basis : bool, optional
            Whether to return the graph in the subgraph basis defined in 
            Dong et al. 2024 ("CktGNN").

        """
        self.subgraph_basis = kwargs.pop("subgraph_basis", False)
        super(CktDataset_igraph, self).__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        return self.graphs[idx][1 - int(self.subgraph_basis)], self.targets[idx]
