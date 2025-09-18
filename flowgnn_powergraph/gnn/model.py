"""
GNN models, How it is structured and types of GNN models: Transformer, GAT, GCN, GIN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, GATv2Conv, NNConv, GINEConv, TransformerConv, SAGEConv, GPSConv, ResGatedGraphConv
from torch_geometric.data.batch import Batch
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention
from typing import Optional

from gnn.conv import FlowGATConv, FlowGATv2Conv, FlowTransformerConv
from gnn.sat.models import GraphTransformer as StructureAwareTransformer
from gnn.exphormer.model import ExphormerModel
from utils.parser_utils import fix_random_seed

from torch_geometric.data import Data


def get_gnnNets(input_dim, output_dim, model_params, graph_regression):
    if model_params["model_name"].lower() in [
        "base",
        "gcn",
        "graphsage",
        "gin",
        "gine",
        "gat",
        "flowgat",
        "gatv2",
        "flowgatv2",
        "transformer",
        "flowtransformer",
        "graphgps",
        "sat",
        "exphormer"
    ]:
        GNNmodel = model_params["model_name"].upper()
        return eval(GNNmodel)(
            input_dim=input_dim, output_dim=output_dim, model_params=model_params, graph_regression=graph_regression
        )
    else:
        raise ValueError(
            f"GNN name should be gcn, gat, gin or transformer " f"and {model_params['model_name']} is not defined."
        )


def identity(x: torch.Tensor, batch: torch.Tensor):
    return x


def cat_max_sum(x, batch):
    node_dim = x.shape[-1]
    bs = max(torch.unique(batch)) + 1
    num_node = int(x.shape[0] / bs)
    x = x.reshape(-1, num_node, node_dim)
    return torch.cat([x.max(dim=1)[0], x.sum(dim=1)], dim=-1)


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
        "identity": identity,
        "cat_max_sum": cat_max_sum,
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    return readout_func_dict[readout.lower()]


class GNNPool(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = get_readout_layers(readout)

    def forward(self, x, batch):
        return self.readout(x, batch)


##
# GNN models
##
class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r"""Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                else:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if hasattr(data, "edge_weight") and data.edge_weight is not None:
                    edge_weight = data.edge_weight
                else:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_attr = torch.ones(
                    (edge_index.shape[1], self.edge_dim),
                    dtype=torch.float32,
                    device=x.device,
                )
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            elif len(args) == 3:
                x, edge_index, edge_attr = args[0], args[1], args[2]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            elif len(args) == 4:
                x, edge_index, edge_attr, batch = args[0], args[1], args[2], args[3]
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            else:
                raise ValueError(
                    f"forward's args should take 1, 2 or 3 arguments but got {len(args)}"
                )
        else:
            data: Batch = kwargs.get("data")
            if not data:
                x = kwargs.get("x")
                edge_index = kwargs.get("edge_index")
                adj = kwargs.get("adj")
                edge_weight = kwargs.get("edge_weight")
                if "edge_index" not in kwargs:
                    assert (
                        adj is not None
                    ), "forward's args is empty and required adj is not in kwargs"
                    if torch.is_tensor(adj):
                        edge_index, edge_weight = from_adj_to_edge_index_torch(adj)
                    else:
                        edge_index, edge_weight = from_adj_to_edge_index_torch(
                            torch.from_numpy(adj)
                        )
                if "adj" not in kwargs:
                    assert (
                        edge_index is not None
                    ), "forward's args is empty and required edge_index is not in kwargs"
                assert (
                    x is not None
                ), "forward's args is empty and required node features x is not in kwargs"
                edge_attr = kwargs.get("edge_attr")
                if "edge_attr" not in kwargs:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                batch = kwargs.get("batch")
                if torch.is_tensor(batch):
                    if batch.size == 0:
                        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                else:
                    if not batch:
                        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if "edge_weight" not in kwargs:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)

            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                    if edge_attr is None:
                        edge_attr = torch.ones(
                            (edge_index.shape[1], self.edge_dim),
                            dtype=torch.float64,
                            device=x.device,
                        )
                else:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                    if batch is None:
                        batch = torch.zeros(
                            x.shape[0], dtype=torch.int64, device=x.device
                        )
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if hasattr(data, "edge_weight"):
                    edge_weight = data.edge_weight
                    if edge_weight is None:
                        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
                else:
                    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
        return x, edge_index, edge_attr, edge_weight, batch


# Basic structure of GNNs
class GNN_basic(GNNBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression,
    ):
        super(GNN_basic, self).__init__()#edge_dim)
        fix_random_seed(model_params["seed"])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = model_params["edge_dim"]
        self.num_layers = model_params["num_layers"]
        self.hidden_dim = model_params["hidden_dim"]
        self.dropout = model_params["dropout"]
        # readout
        self.readout = model_params["readout"]
        self.readout_layer = GNNPool(self.readout)
        #self.default_num_nodes = model_params["default_num_nodes"]
        self.get_layers()
        self.graph_regression = graph_regression
        self.pe_transform = None

    def get_layers(self):
        # GNN layers
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(NNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return

    def forward(self, *args, **kwargs):
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            return self.probs

    def loss(self, pred, label):
        if self.graph_regression:
            return F.mse_loss(pred, label)
        else:
            return F.cross_entropy(pred, label)

    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index, edge_attr*edge_weight[:, None])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def get_graph_rep(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, batch = self._argsparse(*args, **kwargs)
        for layer in self.convs:
            x = layer(x, edge_index, edge_attr*edge_weight[:, None])
            x = F.relu(x) # maybe replace the ReLU with LeakyReLU
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout_layer(x, batch)
        return x

    def get_pred_label(self, pred):
        return pred.argmax(dim=1)


class GAT(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params, graph_regression):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GATConv(current_dim, self.hidden_dim, edge_dim=self.edge_dim, concat=False)
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        alphas = []
        for layer in self.convs:
            x, alpha = layer(x, edge_index, edge_attr*edge_weight[:, None], return_attention_weights=True)
            alphas.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, alphas
    
    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.pop("return_attention_weights", False)
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, alpha = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            if return_attention_weights:
                return self.logits, alpha
            else:
                return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            if return_attention_weights:
                return self.probs, alpha
            else:
                return self.probs
    

class GATV2(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params, graph_regression):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GATv2Conv(current_dim, self.hidden_dim, edge_dim=self.edge_dim, concat=False)
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        alphas = []
        for layer in self.convs:
            x, alpha = layer(x, edge_index, edge_attr*edge_weight[:, None], return_attention_weights=True)
            alphas.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, alphas
    
    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.pop("return_attention_weights", False)
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, alpha = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            if return_attention_weights:
                return self.logits, alpha
            else:
                return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            if return_attention_weights:
                return self.probs, alpha
            else:
                return self.probs


class FLOWGAT(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params, graph_regression):
        self.heads = model_params["heads"]
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                FlowGATConv(current_dim, self.hidden_dim, edge_dim=self.edge_dim, concat=False)
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        alphas = []
        for layer in self.convs:
            x, alpha = layer(x, edge_index, edge_attr*edge_weight[:, None], return_attention_weights=True)
            alphas.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, alphas
    
    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.pop("return_attention_weights", False)
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, alpha = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            if return_attention_weights:
                return self.logits, alpha
            else:
                return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            if return_attention_weights:
                return self.probs, alpha
            else:
                return self.probs
    

class FLOWGATV2(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params, graph_regression):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                FlowGATv2Conv(current_dim, self.hidden_dim, edge_dim=self.edge_dim, concat=False)
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        alphas = []
        for layer in self.convs:
            x, alpha = layer(x, edge_index, edge_attr*edge_weight[:, None], return_attention_weights=True)
            alphas.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, alphas
    
    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.pop("return_attention_weights", False)
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, alpha = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            if return_attention_weights:
                return self.logits, alpha
            else:
                return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            if return_attention_weights:
                return self.probs, alpha
            else:
                return self.probs
    


class GCN(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(GCNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, _, _, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    

class GRAPHSAGE(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(SAGEConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, _, _, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class GIN(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GINConv(
                    nn=nn.Sequential(
                        nn.Linear(current_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                )
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, _, _, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x
    

class GINE(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(current_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    ),
                    edge_dim=self.edge_dim,
                )
            )
            current_dim = self.hidden_dim
        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        for layer in self.convs:
            x = layer(x, edge_index, edge_attr*edge_weight[:, None])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class TRANSFORMER(GNN_basic): #uppercase
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                   TransformerConv(current_dim, self.hidden_dim, heads=4, edge_dim=self.edge_dim, concat=False)
                   )
            current_dim = self.hidden_dim

        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        alphas = []
        for layer in self.convs:
            x, alpha = layer(x, edge_index, edge_attr*edge_weight[:, None], return_attention_weights=True)
            alphas.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, alphas
    
    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.pop("return_attention_weights", False)
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, alpha = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            if return_attention_weights:
                return self.logits, alpha
            else:
                return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            if return_attention_weights:
                return self.probs, alpha
            else:
                return self.probs
    

class FLOWTRANSFORMER(GNN_basic): #uppercase
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                   FlowTransformerConv(current_dim, self.hidden_dim, heads=4, edge_dim=self.edge_dim, concat=False)
                   )
            current_dim = self.hidden_dim

        # FC layers
        mlp_dim = current_dim*2 if self.readout=='cat_max_sum' else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return
    
    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)

        alphas = []
        for layer in self.convs:
            x, alpha = layer(x, edge_index, edge_attr*edge_weight[:, None], return_attention_weights=True)
            alphas.append(alpha)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, alphas
    
    def forward(self, *args, **kwargs):
        return_attention_weights = kwargs.pop("return_attention_weights", False)
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, alpha = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            if return_attention_weights:
                return self.logits, alpha
            else:
                return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            if return_attention_weights:
                return self.probs, alpha
            else:
                return self.probs
            


class GRAPHGPS(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

        self.pe_dim = self.hidden_dim // 8
        self.node_emb = nn.Linear(input_dim - 20, self.hidden_dim - self.pe_dim)
        self.pe_lin = nn.Linear(20, self.pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)

    def get_layers(self):
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            mpnn = ResGatedGraphConv( # GatedGCN
                self.hidden_dim, self.hidden_dim, act=nn.ReLU(), 
                edge_dim=self.edge_dim
            )
            conv = GPSConv(
                self.hidden_dim, mpnn, heads=4, 
                attn_type="multihead", attn_kwargs={"dropout": 0.5}
            )
            self.convs.append(conv)

        # FC layers
        mlp_dim = self.hidden_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=None
        )
        return
    
    def get_emb(self, data, **kwargs):
        x, edge_index, edge_attr, edge_weight, batch = self._argsparse(data, **kwargs)
        x_node = self.node_emb(x[:, :3].squeeze(-1))
        x_pe = self.pe_lin(self.pe_norm(x[:, -20:]))
        x = torch.cat((x_node, x_pe), 1)

        for layer in self.convs:
            x = layer(x, edge_index, batch, edge_attr=edge_attr*edge_weight[:, None])
            x = F.dropout(x, self.dropout, training=self.training)

        return x
    
    def forward(self, data, **kwargs):
        _, _, _, _, batch = self._argsparse(data, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(data, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            return self.probs
    

class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class SAT(StructureAwareTransformer):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression
    ):
        super().__init__(
            in_size=input_dim,
            num_class=output_dim,
            d_model=model_params["hidden_dim"],
            dim_feedforward=2*model_params["hidden_dim"],
            num_heads=4,
            num_layers=model_params["num_layers"],
            batch_norm=True,
            gnn_type="pna",
            use_edge_attr=True,
            num_edge_features=model_params["edge_dim"],
            edge_dim=model_params["edge_dim"],
            k_hop=3,
            se="gnn",
            global_pool="add",
            in_embed=False,
            edge_embed=False,
            deg=model_params.pop("deg", False),
        )
        self.graph_regression = graph_regression

    def forward(self, *args, **kwargs):
        self.logits = super().forward(*args, **kwargs)
        if self.graph_regression:
            return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            return self.probs
        

class EXPHORMER(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
        graph_regression
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
            graph_regression,
        )

        self.pe_dim = self.hidden_dim // 8
        self.node_emb = nn.Linear(input_dim - 20, self.hidden_dim - self.pe_dim)
        self.edge_emb = nn.Linear(self.edge_dim, self.hidden_dim)
        self.pe_lin = nn.Linear(20, self.pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)


    def get_layers(self):

        self.convs = ExphormerModel(
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
            num_heads = 4
        )

        # FC layers
        mlp_dim = self.hidden_dim
        self.mlps = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, self.output_dim))
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=None
        )
        return
    
    def get_emb(self, data, **kwargs):
        x, edge_index, edge_attr, edge_weight, batch = self._argsparse(data, **kwargs)
        x_node = self.node_emb(x[:, :3].squeeze(-1))
        x_pe = self.pe_lin(self.pe_norm(x[:, -20:]))
        x = torch.cat((x_node, x_pe), 1)
        edge_attr = self.edge_emb(edge_attr*edge_weight[:, None])

        data = Data(
            x=x, edge_index=edge_index, 
            edge_attr=edge_attr*edge_weight[:, None], batch=batch
        )
        return self.convs(data)
    
    def forward(self, data, **kwargs):
        _, _, _, _, batch = self._argsparse(data, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(data, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            return self.probs