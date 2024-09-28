import torch
import torch.nn as nn
import torch.nn.functional as F

from flowgnn_opamps.original_code.constants import *
from flowgnn_opamps.original_code.dagnn_pyg import DAGNN as OriginalDAGNN
from flowgnn_opamps.original_code.batch import Batch

class DAGNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            max_n, 
            nvt, 
            START_TYPE, 
            END_TYPE, 
            hs=301, 
            nz=66, 
            bidirectional=False, 
            num_layers=2,
            agg=NA_ATTN_H, 
            out_wx=False, 
            out_pool_all=False, 
            out_pool=P_MAX, 
            dropout=0.0,
            num_nodes=8,
            pred_hid_channels = 64,
            pred_dropout=0.5,
            seed=None
        ):
        super(DAGNN, self).__init__()
        if seed:
            torch.manual_seed(seed)

        self.decoder = None
        self.backbone = OriginalDAGNN(emb_dim, hs, hs,
                                      max_n=max_n, nvt=nvt, 
                                      START_TYPE=START_TYPE, 
                                      END_TYPE=END_TYPE, hs=hs, nz=nz, 
                                      num_layers=num_layers, 
                                      bidirectional=bidirectional,
                                      agg=agg, out_wx=out_wx, 
                                      out_pool_all=out_pool_all, 
                                      out_pool=out_pool, dropout=dropout,
                                      num_nodes=num_nodes)
        
        self.fc_pred = nn.Sequential(
            nn.Linear(self.backbone.hs * num_layers, pred_hid_channels),
            nn.ReLU(),
            nn.Linear(pred_hid_channels, 1)
        )
        self.pred_hid_channels = pred_hid_channels
        self.pred_dropout = pred_dropout
    
    def get_device(self):
        return next(self.parameters()).device
    
    def _collate_fn(self, G):
        return self.backbone._collate_fn(G)
    
    def predict(self, g):
        if type(g) != list:
            g = [g]
        # encode graphs G into latent vectors
        b = Batch.from_data_list(g)
        hg = self.backbone(b)
        hg = F.dropout(hg, p=self.pred_dropout, training=self.training)
        return self.fc_pred(hg)
    
    def forward(self, g):
        self.backbone.forward(g)