"""Original code copied from https://github.com/LUOyk1999/DAGformer
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from torch_scatter import scatter_add, scatter_mean
from einops import rearrange

from .utils_dag_transformer import pad_batch, unpad_batch, add_order_info


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
            edge_attr=None, ptr=None, return_attn=False):
        output = x
        for mod in self.layers:
            output = mod(output, SAT, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False,
                 gnn_type="gcn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', SAT=False, **kwargs):
        super().__init__()

        self.SAT = SAT
        self.dropout = nn.Dropout(0.1)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model) 
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, data, return_attn=False):
        data = add_order_info(data)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        mask_dag_ = data.mask_rc if hasattr(data, 'mask_rc') else None
        dag_rr_edge_index = data.dag_rr_edge_index if hasattr(data, 'dag_rr_edge_index') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
        output = self.dropout(output)

        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
        else:
            edge_attr = None

        output = self.encoder(
            output, 
            self.SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )
        # readout step
        if self.use_global_pool:
            output = self.pooling(output, data.batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        return self.classifier(output)
    
class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                activation="relu", batch_norm=True, pre_norm=False,
                gnn_type="gcn", **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
            bias=False, gnn_type=gnn_type, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, SAT, edge_index, mask_dag_, dag_rr_edge_index,
            edge_attr=None, ptr=None,
            return_attn=False,
        ):
        
        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )

        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x
    
class Attention(gnn.MessagePassing):
    """Multi-head DAG attention using PyG interface
    accept Batch data given by PyG
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False, symmetric=False, gnn_type="gcn", **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.gnn_type = gnn_type
        self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
            self.to_tqk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

        self.batch_first = False

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x,
            SAT,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=None,
            ptr=None,
            return_attn=False):
        
        # Compute value matrix

        v = self.to_v(x)
        
        if(SAT):
            # using DAG+SAT
            x_struct = self.structure_extractor(x, edge_index, edge_attr)
        else:
            # using DAG+transformer
            x_struct = x
        
        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)
        
        # Compute self-attention
        attn = None
        if dag_rr_edge_index is not None:
            # print(dag_rr_edge_index.shape)
            out = self.propagate(dag_rr_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    dag_rr_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)
            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, mask_dag_, return_attn=return_attn)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention based on MPNN """
        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, mask_dag_, return_attn=False):
        """ Self attention based on mask matrix""" 
       
        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        # DAG mask
        mask_dag_ = mask_dag_.reshape(dots.shape[0],mask_dag_.shape[1],mask_dag_.shape[1])
        mask_dag_ = mask_dag_[:,:dots.shape[2],:dots.shape[3]]
        dots = dots.masked_fill(
            mask_dag_.unsqueeze(1),
            float('-inf'),
        )
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None
    

class StructureExtractor(nn.Module):

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x = self.relu(gcn_layer(x, edge_index))
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x
    
EDGE_GNN_TYPES = [
    'gine', 'gcn',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4'
]


def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    edge_dim = kwargs.get('edge_dim', None)
    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, embed_dim)
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, embed_dim)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        else:
            return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gin":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    elif gnn_type == "gine":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    elif gnn_type == "pna":
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna2":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                           aggregators=aggregators, scalers=scalers,
                           deg=deg, towers=4, pre_layers=1, post_layers=1,
                           divide_input=True, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna3":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna4":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=8, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer


    elif gnn_type == "mpnn":
        aggregators = ['sum']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    else:
        raise ValueError("Not implemented!")
    
class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = utils.degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out