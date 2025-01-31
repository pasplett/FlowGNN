import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    MLP,
    GCNConv,
    GINConv,
    GATConv,
    GATv2Conv,
    TransformerConv,
    global_mean_pool
)
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.models import MLP

from flowgnn_opamps.original_code.constants import *
from flowgnn_opamps.original_code.models_ig import DVAE
from flowgnn_opamps.original_code.pace import PACE_VAE
from flowgnn_opamps.conv import FlowGATConv, FlowGATv2Conv, FlowTransformerConv

class BaseRegression(nn.Module):
    def __init__(
            self, 
            in_channels, 
            hidden_channels, 
            num_layers, 
            pred_hidden_channels=64,
            pred_dropout=0.5,
            undirected=False,
            **kwargs):
        super(BaseRegression, self).__init__()

        seed = kwargs.pop("seed", None)
        if seed:
            torch.manual_seed(seed)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.fc_pred = nn.Sequential(
            nn.Linear(hidden_channels, pred_hidden_channels),
            nn.ReLU(),
            nn.Linear(pred_hidden_channels, 1)
        )
        self.pred_hidden_channels = pred_hidden_channels
        self.pred_dropout = pred_dropout
        self.undirected = undirected

    def _collate_fn(self, G):
        return G

    def get_device(self):
        return next(self.parameters()).device
    
    def _to_undirected(self, edge_index):
        return torch.cat([edge_index, torch.flip(edge_index, dims=(0,))], dim=1)
    
    def forward(self, g):
        x, edge_index, batch = g.x, g.edge_index, g.batch
        if self.undirected:
            edge_index = self._to_undirected(edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.pred_dropout, training=self.training)
        return self.fc_pred(x)
    
    def predict(self, g):
        return self(g)
    
    def encode(self, g):
        x, edge_index, batch = g.x, g.edge_index, g.batch
        if self.undirected:
            edge_index = self._to_undirected(edge_index)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return x



class GCN(BaseRegression):
    def __init__(self, in_channels, hidden_channels, num_layers, **kwargs):
        super(GCN, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels


class GIN(BaseRegression):
    def __init__(self, in_channels, hidden_channels, num_layers, **kwargs):
        super(GIN, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels


class GAT(BaseRegression):
    def __init__(
            self, in_channels, hidden_channels, num_layers, heads=1, **kwargs
        ):
        super(GAT, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads)
            )
            in_channels = hidden_channels * heads
        self.convs.append(GATConv(in_channels, hidden_channels, heads=1))


class GATv2(BaseRegression):
    def __init__(
            self, in_channels, hidden_channels, num_layers, heads=1, **kwargs
        ):
        super(GATv2, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(in_channels, hidden_channels, heads=heads)
            )
            in_channels = hidden_channels * heads
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=1))


class Transformer(BaseRegression):
    def __init__(
            self, in_channels, hidden_channels, num_layers, heads=1, **kwargs
        ):
        super(Transformer, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
            )
            in_channels = hidden_channels


class FlowGAT(BaseRegression):
    def __init__(
            self, in_channels, hidden_channels, num_layers, heads=1, **kwargs
        ):
        super(FlowGAT, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                FlowGATConv(in_channels, hidden_channels, heads=heads)
            )
            in_channels = hidden_channels * heads
        self.convs.append(FlowGATConv(in_channels, hidden_channels, heads=1))


class FlowGATv2(BaseRegression):
    def __init__(
            self, in_channels, hidden_channels, num_layers, heads=1, **kwargs
        ):
        super(FlowGATv2, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                FlowGATv2Conv(in_channels, hidden_channels, heads=heads)
            )
            in_channels = hidden_channels * heads
        self.convs.append(FlowGATv2Conv(in_channels, hidden_channels, heads=1))


class FlowTransformer(BaseRegression):
    def __init__(
            self, in_channels, hidden_channels, num_layers, heads=1, **kwargs
        ):
        super(FlowTransformer, self).__init__(
            in_channels, hidden_channels, num_layers, **kwargs
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                FlowTransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
            )
            in_channels = hidden_channels


class DVAE_Regression(DVAE):
    def __init__(
            self,
            max_n, 
            nvt, 
            feat_nvt, 
            START_TYPE, 
            END_TYPE, 
            hs=301, 
            nz=66, 
            bidirectional=False, 
            vid=False, 
            max_pos=8, 
            topo_feat_scale=0.01, 
            pred_hidden_channels=64, 
            pred_dropout=0.5, 
            seed=None
        ):
        
        if seed:
            torch.manual_seed(seed)

        super(DVAE_Regression, self).__init__(max_n, nvt, feat_nvt, START_TYPE, 
                                              END_TYPE, hs=hs, nz=nz, 
                                              bidirectional=bidirectional, 
                                              vid=vid, max_pos=max_pos, 
                                              topo_feat_scale=topo_feat_scale)

        self.fc_pred = nn.Sequential(
            nn.Linear(self.gs, pred_hidden_channels),
            nn.ReLU(),
            nn.Linear(pred_hidden_channels, 1)
        )
        self.pred_hidden_channels = pred_hidden_channels
        self.pred_dropout = pred_dropout
    
    def predict(self, g):
        # encode graphs G into latent vectors
        if type(g) != list:
            g = [g]
        self._propagate_from(g, 0, self.grue_forward, H0=self._get_zero_hidden(len(g)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(g, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(g)), reverse=True)
        hg = self._get_graph_state(g)
        hg = F.dropout(hg, p=self.pred_dropout, training=self.training)
        return self.fc_pred(hg)
    
    def encode(self, g):
        # encode graphs G into latent vectors
        if type(g) != list:
            g = [g]
        self._propagate_from(g, 0, self.grue_forward, H0=self._get_zero_hidden(len(g)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(g, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(g)), reverse=True)
        hg = self._get_graph_state(g)
        return hg
    

class PACE_Regression(PACE_VAE):
    def __init__(
            self,
            max_n, 
            nvt, 
            START_TYPE, 
            END_TYPE, 
            ninp=256, 
            nhead=6, 
            nhid=512, 
            nlayers=5, 
            dropout=0.25, 
            fc_hidden=256, 
            nz = 64,
            pred_hidden_channels=64, 
            pred_dropout=0.5, 
            seed=None
        ):
        
        if seed:
            torch.manual_seed(seed)

        super(PACE_Regression, self).__init__(
            max_n, nvt, START_TYPE, END_TYPE, 0, ninp=ninp, 
            nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout, 
            fc_hidden=fc_hidden, nz=nz
        )

        self.fc_pred = nn.Sequential(
            nn.Linear(self.hidden_size, pred_hidden_channels),
            nn.ReLU(),
            nn.Linear(pred_hidden_channels, 1)
        )
        self.pred_hidden_channels = pred_hidden_channels
        self.pred_dropout = pred_dropout
    
    def get_device(self):
        return next(self.parameters()).device
    
    def predict(self, g):
        memory = self.encode(g)
        memory = F.dropout(memory, p=self.pred_dropout, training=self.training)
        return self.fc_pred(memory)
    
    def encode(self, glist):
        node_one_hot, pos_one_hot, adj, src_mask, tgt_mask, mem_mask, graph_sizes, true_types = self._prepare_features(glist)
        bsize = len(graph_sizes) 

        pos_feat = self.pos_embed(pos_one_hot, adj)
        node_feat = self.node_embed(node_one_hot)
        node_feat = torch.cat([node_feat,pos_feat],2)

        # here we set the source sequence and the tgt sequence for the teacher forcing
        src_inp = node_feat.transpose(0,1) # node 2 is the start symbol, shape: (bsiaze, max_n-1, nhid)

        
        #memory = self.encoder(src_inp,mask=src_mask)
        memory = self.encoder(src_inp,mask=tgt_mask)
        memory = memory.transpose(0,1).reshape(-1,self.max_n*self.nhid) # shape ( bsize, self.max_n-1, nhid): each batch, the first num_node - 1 rows are the representation of input nodes.
        return memory


class FlowPredictor(nn.Module):

    def __init__(self, in_channels_succ, in_channels_v, rev=False):
        super(FlowPredictor, self).__init__()
        self.fc_succ = nn.Linear(in_channels_succ, 1)
        self.fc_v = nn.Linear(in_channels_v, 1)
        self.softmax = nn.Softmax(dim=1)

    def _softmax_ignore_zeros(self, x):
        """
        Zero input into linear layers (due to shape matching for batch_size > 1) 
        should result in zero attention weights. Therefore, the linear layer's
        output is set to a large negative value in this case.
        """
        x[x == 0] = -1e30
        return x

    def forward(self, x, h, A):
        h_succ, h_v = self.fc_succ(x), self.fc_v(h) # (num_nodes, 1)
        att_weights = h_succ + h_v.T # (num_nodes, num_nodes)ß
        att_weights = self._softmax_ignore_zeros(
            A * att_weights
        ) # (num_nodes, num_nodes)
        flow = A * self.softmax(att_weights) # (num_nodes, num_nodes) 
        return flow


class AttentionAggregation(nn.Module):

    def __init__(self, in_channels_v, in_channels_pred):
        super(AttentionAggregation, self).__init__()
        self.fc_v = nn.Linear(in_channels_v, 1)
        self.fc_pred = nn.Linear(in_channels_pred, 1)
        self.softmax = nn.Softmax(dim=0)

    def _softmax_ignore_zeros(self, x):
        """
        Zero input into linear layers (due to shape matching for batch_size > 1) 
        should result in zero attention weights. Therefore, the linear layer's
        output is set to a large negative value in this case.
        """
        x[x == 0] = -1e30
        return x

    def forward(self, x, h, A):
        h_v, h_pred = self.fc_v(x), self.fc_pred(h) # (num_nodes, 1)
        att_weights = h_v.T + h_pred # (num_nodes, num_nodes)
        att_weights = self._softmax_ignore_zeros(
            A * att_weights
        ) # (num_nodes, num_nodes)
        att_weights = A * self.softmax(att_weights) # (num_nodes, num_nodes) 
        return att_weights

class FlowDAGNN(nn.Module):

    def __init__(
            self, 
            n_types=10,
            n_feats=1, 
            hid_channels=301, 
            pred_hid_channels=64,
            pred_out_channels=1,
            num_layers=2,
            dropout=0.5,
            **kwargs
        ):
        super(FlowDAGNN, self).__init__()

        seed = kwargs.pop("seed", None)
        if seed:
            torch.manual_seed(seed)
        self.device = kwargs.pop("device", None)

        self.n_types = n_types
        self.n_feats = n_feats
        in_channels = n_types + n_feats
        self.in_channels = in_channels
        self.hid_channels = hid_channels      
        self.num_layers = num_layers
        self.dropout = dropout

        # Modules     
        att_aggr = []
        flow_predictors = []
        enc_gru_cells_forward = []
        enc_gru_cells_backward = []
        for l in range(self.num_layers):
            att_aggr.append(
                AttentionAggregation(in_channels, hid_channels)
            )
            flow_predictors.append(
                FlowPredictor(hid_channels, hid_channels)
            )
            enc_gru_cells_forward.append(
                nn.GRUCell(hid_channels, hid_channels)
            )
            enc_gru_cells_backward.append(
                nn.GRUCell(in_channels, hid_channels)
            )
            in_channels = hid_channels
        self.att_aggr= nn.ModuleList(att_aggr)
        self.flow_predictors = nn.ModuleList(flow_predictors)
        self.enc_gru_cells_forward = nn.ModuleList(enc_gru_cells_forward)
        self.enc_gru_cells_backward = nn.ModuleList(enc_gru_cells_backward)

        # Prediction
        self.fc_pred = nn.Sequential(
            nn.Linear(num_layers * hid_channels * 2, pred_hid_channels),
            nn.ReLU(),
            nn.Linear(pred_hid_channels, pred_out_channels)
        ) 


    def get_device(self):
        return next(self.parameters()).device


    def _check_topological_ordering(self, edge_index):
        assert torch.all(edge_index[0] < edge_index[1])


    def _to_dense(self, edge_index, num_nodes, device=None):
        if device is None:
            device = self.get_device()
        A = torch.zeros((num_nodes, num_nodes), dtype=int, device=device)
        if edge_index.shape[1] > 0:
            # Insert edges
            n_con_nodes = torch.max(edge_index) + 1
            A[:n_con_nodes, :n_con_nodes] = to_dense_adj(edge_index)[0]
        return A


    def _conv(self, x_bw, x_fw, edge_index, batch, layer, m0=None, device=None):
        if device is None:
            device = self.get_device()
        
        # Adjacency matrix
        total_num_nodes = x_fw.shape[0]
        A = self._to_dense(edge_index, total_num_nodes)

        # Initialize att_weights, hidden state and initial message
        att_weights = torch.zeros((total_num_nodes, total_num_nodes), device=device)
        flow = torch.zeros((total_num_nodes, total_num_nodes), device=device)
        h_fw = torch.zeros((x_fw.shape[0], self.hid_channels), device=device)
        h_bw = torch.zeros((x_bw.shape[0], self.hid_channels), device=device)
        if m0 is None:
            m0 = torch.zeros((total_num_nodes, self.hid_channels), device=device)

        # Find positions of source and target nodes as well as 
        # number of nodes per graph in batch
        sources = torch.cat(
            [torch.arange(total_num_nodes, device=device)[batch == i][:1] 
             for i in torch.unique(batch)]
        )
        targets = torch.cat(
            [torch.arange(total_num_nodes, device=device)[batch == i][-1:]
             for i in torch.unique(batch)]
        )
        num_nodes_per_graph = targets - sources + 1
        max_nodes = torch.max(num_nodes_per_graph)

        # BACKWARD PASS
        for i in range(max_nodes):
            vs = targets - i
            vs = vs[vs >= sources]
            vs = F.one_hot(vs, total_num_nodes).sum(dim=0).reshape((total_num_nodes, 1))

            # Aggregate
            att_weights = torch.clone(att_weights) + self.att_aggr[layer](
                x_bw, h_bw, (A * vs).T
            )
            if i == 0:
                message = m0
            else:
                message = torch.matmul(att_weights.T, h_bw) * vs

            # Update
            h_bw = torch.clone(h_bw) + self.enc_gru_cells_backward[layer](x_bw, message) * vs

        # FORWARD PASS
        for i in range(max_nodes):

            vs = sources + i
            vs = vs[i < num_nodes_per_graph]
            vs = F.one_hot(vs, total_num_nodes).sum(dim=0).reshape((total_num_nodes, 1))

            # Aggregate
            if i == 0:
                message = h_bw * vs
            else:
                message = torch.matmul(flow.T, h_fw) * vs

            # Combine
            h_fw = torch.clone(h_fw) + self.enc_gru_cells_forward[layer](h_bw, message) * vs

            # Flow update
            if i < max_nodes - 1:
                flow = torch.clone(flow) + self.flow_predictors[layer](
                    h_bw, h_fw, A * vs
                )

        return h_bw, h_fw, att_weights, flow
    

    def _graph_embedding(self, x, edge_index, batch, m0=None, 
                         return_vertex_states=False, device=None):

        # Check topological ordering
        self._check_topological_ordering(edge_index)
        
        if device is None:
            device = self.get_device()

        # Preparation
        num_nodes = x.shape[0]
        sources = torch.cat(
            [torch.arange(num_nodes, device=device)[batch == i][:1] 
             for i in torch.unique(batch)]
        )
        targets = torch.cat(
            [torch.arange(num_nodes, device=device)[batch == i][-1:] 
             for i in torch.unique(batch)]
        )
        batch_size = torch.max(batch) + 1
        h_out = torch.zeros((batch_size, 2 * self.num_layers * self.hid_channels), 
                            device=device)
        flow = torch.zeros((self.num_layers, num_nodes, num_nodes), 
                           device=device)
        att_weights = torch.zeros((self.num_layers, num_nodes, num_nodes), 
                           device=device)

        # Forward loop
        x_bw, x_fw = torch.clone(x), torch.clone(x)
        for layer in range(self.num_layers):
            x_bw, x_fw, att_w_layer, flow_layer = self._conv(
                x_bw, x_fw, edge_index, batch, layer, m0=m0, device=device
            )
            h_out[
                :, layer * 2 * self.hid_channels : (layer + 1) * 2 * self.hid_channels
            ] = torch.cat([x_bw[sources], x_fw[targets]], dim=1)
            att_weights[layer] = att_w_layer
            flow[layer] = flow_layer

        if return_vertex_states:
            return h_out, att_weights, flow, x_fw
        return h_out, att_weights, flow

    def predict(self, g, return_flow=False):
        x, edge_index, batch = g.x, g.edge_index, g.batch
        hg, att_weights, flow = self._graph_embedding(x, edge_index, batch)
        hg = F.dropout(hg, p=self.dropout, training=self.training)
        if return_flow:
            return self.fc_pred(hg), att_weights, flow
        else:
            return self.fc_pred(hg)
    
    def forward(self, g):
        return self.predict(g)
    
    def encode(self, g):
        x, edge_index, batch = g.x, g.edge_index, g.batch
        hg, _ = self._graph_embedding(x, edge_index, batch)
        return hg