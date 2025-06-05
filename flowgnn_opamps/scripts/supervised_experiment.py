import numpy as np
import os
import pandas as pd
import time
import torch

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from flowgnn_opamps.regression_models import (
    GCN,
    GIN,
    GAT,
    GATv2,
    Transformer,
    FlowGAT,
    FlowGATv2,
    FlowTransformer,
    DVAE_Regression,
    PACE_Regression,
    DAG_Transformer,
    FlowDAGNN,
)
from flowgnn_opamps.dataset import CktDataset, CktDataset_DAGFormer, CktDataset_igraph
from flowgnn_opamps.utils import (
    train_pygraph_supervised,
    train_igraph_supervised,
    test_pygraph_supervised,
    test_igraph_supervised
)
from flowgnn_opamps.callbacks import EarlyStoppingCB
from flowgnn_opamps.original_code.constants import *

# Check GPU availability
gpu_available = torch.cuda.is_available()

if gpu_available:
    print("Running on GPU")
    device = "cuda:0"
else:
    print("Running on CPU")
    device = "cpu"

# Pathes to data
pygraph_file = "./../data/ckt_bench_101_tg26.pkl" # Graph data
igraph_file = "./../data/ckt_bench_101.pkl" # Graph data
target_file = "./../data/perform101.csv" # Target properties

# Experimental setup
num_runs = 10
start_run = 0
seeds = [4, 42, 420, 4204, 42042, 420420, 4204204, 42042042, 420420420, 4204204204]
model_names = ["flowgat", "flowgatv2", "flowtransformer", "flowdagnn"]
properties = ["gain", "bw", "fom"]

# Training hyperparameters
batch_size = 64
infer_batch_size = 128
num_epochs = 1
learning_rate = 0.0001
save_interval = 1000
pred_hidden_channels = 64
pred_dropout = 0.5
patience = 20
loss_function = torch.nn.MSELoss()
lr_scheduler = False
#DAGFormer
use_mpnn = False

model_parameters = {
    "gcn": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "gin": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "gat": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "heads": 8,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    
    "gatv2": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "heads": 8,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "transformer": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "heads": 8,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "flowgat": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "heads": 8,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "flowgatv2": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "heads": 8,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "flowtransformer": {
        "in_channels": 11,
        "hidden_channels": 301,
        "num_layers": 2,
        "heads": 8,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "undirected": False
    },
    "dvae": {
        "max_n": 24,
        "nvt": 10,
        "feat_nvt": 1,
        "START_TYPE": 8,
        "END_TYPE": 9,
        "hs": 301,
        "nz": 66,
        "bidirectional": False,
        "vid": False,
        "max_pos": 8,
        "topo_feat_scale": 0.01,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout
    },
    "dvae_bidir": {
        "max_n": 24,
        "nvt": 10,
        "feat_nvt": 1,
        "START_TYPE": 8,
        "END_TYPE": 9,
        "hs": 301,
        "nz": 66,
        "bidirectional": True,
        "vid": False,
        "max_pos": 8,
        "topo_feat_scale": 0.01,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout
    },
    "flowdagnn": {
        "n_types": 10,
        "n_feats": 1,
        "hid_channels": 301,
        "pred_hid_channels": pred_hidden_channels,
        "pred_out_channels": 1,
        "num_layers": 2,
        "dropout": pred_dropout
    },
    "pace": {
        "max_n": 24, 
        "nvt": 10,
        "START_TYPE": 8, 
        "END_TYPE": 9,
        "ninp": 256, 
        "nhead": 8, 
        "nhid": 512, 
        "nlayers": 6, 
        "dropout": 0.25, 
        "fc_hidden": 256, 
        "nz": 66
    },
    "dag_transformer": {
        "in_size": 11, 
        "d_model": 256, 
        "num_heads": 8,
        "dim_feedforward": 512, 
        "dropout": 0.2, 
        "num_layers": 4,
        "batch_norm": True,
        "gnn_type": "gcn", 
        "use_edge_attr": True, 
        "num_edge_features": 4,
        "in_embed": False, 
        "edge_embed": False, 
        "use_global_pool": True,
        "global_pool": 'mean', 
        "SAT": True,
    }
}

res_dir = "./../experiments/supervised/"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Experiment
for i, model_name in enumerate(model_names):
    
    print(f"Model = {model_name.upper()}\n")

    # Create model directory and save hyperparameters
    model_dir = res_dir + f"{model_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    hyperparameters = {
        "batch_size": batch_size,
        "infer_batch_size": infer_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "pred_hidden_channels": pred_hidden_channels,
        "pred_dropout": pred_dropout,
        "patience": patience,
        "lr_scheduler": lr_scheduler,
        "seeds": seeds,
        f"{model_name}_parameters": model_parameters[model_name]
    }
    torch.save(hyperparameters, model_dir + "hyperparameters.pt")

    
    for j, target in enumerate(properties):

        # Create property directory in model directory
        print(f"Predicting {target}\n")
        prop_dir = model_dir + f"{target}/"
        if not os.path.exists(prop_dir):
            os.makedirs(prop_dir)


        for k in range(start_run, start_run + num_runs):

            torch.manual_seed(seeds[k])

            # Create run directory in property directory
            print(f"Training Run No. {k + 1}/{start_run + num_runs}\n")
            run_dir = prop_dir + f"run_{k + 1}/"
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            
            # Select model
            if model_name.startswith("gcn"):
                model = GCN(**model_parameters[model_name])
                data_mode = "pygraph"
                
            elif model_name.startswith("gin"):
                model = GIN(**model_parameters[model_name])
                data_mode = "pygraph"
            
            elif model_name.startswith("gatv2"):
                model = GATv2(**model_parameters[model_name])
                data_mode = "pygraph"
            
            elif model_name.startswith("gat"):
                model = GAT(**model_parameters[model_name])
                data_mode = "pygraph"

            elif model_name.startswith("transformer"):
                model = Transformer(**model_parameters[model_name])
                data_mode = "pygraph"    
            
            elif model_name.startswith("flowgatv2"):
                model = FlowGATv2(**model_parameters[model_name])
                data_mode = "pygraph"

            elif model_name.startswith("flowgat"):
                model = FlowGAT(**model_parameters[model_name])
                data_mode = "pygraph"

            elif model_name.startswith("flowtransformer"):
                model = FlowTransformer(**model_parameters[model_name])
                data_mode = "pygraph"
                
            elif model_name.startswith("dvae"):
                model = DVAE_Regression(**model_parameters[model_name])
                data_mode = "igraph"

            elif model_name.startswith("pace"):
                model = PACE_Regression(**model_parameters[model_name])
                data_mode = "igraph"

            elif model_name.startswith("dag_transformer"):
                model = DAG_Transformer(**model_parameters[model_name])
                data_mode = "pygraph_dagformer"
            
            elif model_name.startswith("flowdagnn"):
                model = FlowDAGNN(**model_parameters[model_name])
                data_mode = "pygraph"
            

            # Load dataset
            if data_mode == "pygraph":
                train_dataset = CktDataset(pygraph_file, target_file, target=target, train=True, val=False, device=device)
                mean_target, std_target = torch.mean(train_dataset.targets), torch.std(train_dataset.targets)
                train_dataset.targets = (train_dataset.targets - mean_target) / std_target
                
                val_dataset = CktDataset(pygraph_file, target_file, target=target, train=True, val=True, device=device)
                val_dataset.targets = (val_dataset.targets - mean_target) / std_target
                
                test_dataset = CktDataset(pygraph_file, target_file, target=target, train=False, device=device)
                test_dataset.targets = (test_dataset.targets - mean_target) / std_target

                # Define loaders
                train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=False, pin_memory=True)
                val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=False, pin_memory=True)
                test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=False, pin_memory=True)


            elif data_mode == "pygraph_dagformer":
                train_dataset = CktDataset_DAGFormer(pygraph_file, target_file, target=target, train=True, val=False, device=device, use_mpnn=use_mpnn)
                mean_target, std_target = torch.mean(train_dataset.targets), torch.std(train_dataset.targets)
                train_dataset.targets = (train_dataset.targets - mean_target) / std_target
                
                val_dataset = CktDataset_DAGFormer(pygraph_file, target_file, target=target, train=True, val=True, device=device, use_mpnn=use_mpnn)
                val_dataset.targets = (val_dataset.targets - mean_target) / std_target
                
                test_dataset = CktDataset_DAGFormer(pygraph_file, target_file, target=target, train=False, device=device, use_mpnn=use_mpnn)
                test_dataset.targets = (test_dataset.targets - mean_target) / std_target

                # Define loaders
                train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=False, pin_memory=True)
                val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=False, pin_memory=True)
                test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=False, pin_memory=True)


            elif data_mode == "igraph":
                train_dataset = CktDataset_igraph(igraph_file, target_file, target=target, train=True, val=False, device=device,
                                                  subgraph_basis = model_name.startswith("cktgnn"))
                mean_target, std_target = torch.mean(train_dataset.targets), torch.std(train_dataset.targets)
                train_dataset.targets = (train_dataset.targets - mean_target) / std_target
                
                val_dataset = CktDataset_igraph(igraph_file, target_file, target=target, train=True, val=True, device=device,
                                                 subgraph_basis = model_name.startswith("cktgnn"))
                val_dataset.targets = (val_dataset.targets - mean_target) / std_target
                
                test_dataset = CktDataset_igraph(igraph_file, target_file, target=target, train=False, device=device,
                                                 subgraph_basis = model_name.startswith("cktgnn"))
                test_dataset.targets = (test_dataset.targets - mean_target) / std_target

            
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            if lr_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=20)
            early_stopping = EarlyStoppingCB(mode="min", patience=patience)

            best = 1e8
            results = np.zeros((num_epochs, 8))

            # Training Loop
            for epoch in range(1, num_epochs + 1):

                
                if data_mode == "pygraph" or data_mode == "pygraph_dagformer":

                    t0 = time.perf_counter()
                    train_pygraph_supervised(model, train_loader, loss_function, optimizer)
                    t1 = time.perf_counter()
                    train_rmse, train_pearson = test_pygraph_supervised(model, train_loader)
                    t2 = time.perf_counter()
                    val_rmse, val_pearson = test_pygraph_supervised(model, val_loader)
                    test_rmse, test_pearson = test_pygraph_supervised(model, test_loader)
                    
                elif data_mode == "igraph":

                    t0 = time.perf_counter()
                    train_igraph_supervised(model, train_dataset, loss_function, optimizer, batch_size=batch_size)
                    t1 = time.perf_counter()
                    train_rmse, train_pearson = test_igraph_supervised(model, train_dataset, batch_size=infer_batch_size)
                    t2 = time.perf_counter()
                    val_rmse, val_pearson = test_igraph_supervised(model, val_dataset, batch_size=infer_batch_size)
                    test_rmse, test_pearson = test_igraph_supervised(model, test_dataset, batch_size=infer_batch_size)
                    
                if lr_scheduler:
                    scheduler.step(train_rmse)
                
                print(f"Epoch {epoch}/{num_epochs} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

                results[epoch - 1, 0] = train_rmse
                results[epoch - 1, 1] = train_pearson
                results[epoch - 1, 2] = val_rmse
                results[epoch - 1, 3] = val_pearson
                results[epoch - 1, 4] = test_rmse
                results[epoch - 1, 5] = test_pearson
                results[epoch - 1, 6] = t1 - t0
                results[epoch - 1, 7] = t2 - t1
                
                pd.DataFrame(
                    results, 
                    columns=["train_rmse", "train_pearson", "val_rmse", "val_pearson", "test_rmse", "test_pearson", "train_time", "test_time"]
                ).to_csv(run_dir + "results.csv")

                if val_rmse < best:

                    model_checkpoint = os.path.join(run_dir, 'best_model.pth')
                    optimizer_checkpoint = os.path.join(run_dir, 'best_optimizer.pth')
                    torch.save(model.state_dict(), model_checkpoint)
                    torch.save(optimizer.state_dict(), optimizer_checkpoint)
                    if lr_scheduler:
                        scheduler_checkpoint = os.path.join(run_dir, 'best_scheduler.pth')
                        torch.save(scheduler.state_dict(), scheduler_checkpoint)
                    best = val_rmse

                if epoch % save_interval == 0:

                    model_checkpoint = os.path.join(run_dir, 'model_checkpoint{}.pth'.format(epoch))
                    optimizer_checkpoint = os.path.join(run_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
                    torch.save(model.state_dict(), model_checkpoint)
                    torch.save(optimizer.state_dict(), optimizer_checkpoint)
                    if lr_scheduler:
                        scheduler_checkpoint = os.path.join(run_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
                        torch.save(scheduler.state_dict(), scheduler_checkpoint)

                if early_stopping(val_rmse):
                    print(f"Early Stopping! Best Val RMSE: {best}")
                    break
            
            print("\n")
