import pandas as pd
import numpy as np
import tqdm
import csv
import torch
from torch_geometric.data import Batch as DataBatch
from tqdm import tqdm
import igraph
import networkx as nx
import random
import pickle
import gzip
from shutil import copy
from copy import deepcopy

def pygraph_to_igraph(pygraph):
    n_v = pygraph.x.shape[0]
    attr_v = pygraph.x
    edge_idxs = pygraph.edge_index
    g = igraph.Graph(directed=True)
    g.add_vertices(n_v)
    for i in range(n_v):
        g.vs[i]['type'] = torch.argmax(attr_v[i,:-1]).item()
        g.vs[i]['feat'] = attr_v[i,-1].item()
    edges = []
    for src, tgt in zip(edge_idxs[1], edge_idxs[0]):
        edges += [(src, tgt)]
    g.add_edges(edges)
    return g


def is_same_DAG(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True


def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, subg=True):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    if subg:
        START_TYPE=0
        END_TYPE=1
    else:
        START_TYPE=8 
        END_TYPE=9
    res = g.is_dag()
    #return res
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1


def is_valid_Circuit(g, subg=True):
    # Check if the given igraph g is a amp circuits
    # first checks whether the circuit topology is a DAG
    # second checks the node type in the main path
    if subg:
        cond1 = is_valid_DAG(g, subg=True)
        cond2 = True
        for v in g.vs:
            pos = v['pos']
            subg_feats = [v['r'], v['c'], v['gm']]
            if pos in [2,3,4]: # i.e. in the main path
                if v['type'] in [8,9]:
                    cond2 = False
        return cond1 and cond2
    else:
        cond1 = is_valid_DAG(g, subg=False)
        cond2 = True
        diameter_path = g.get_diameter(directed=True) #find the main path the diameter path must start/end at the sudo input/end node
        if len(diameter_path) < 3:
            cond2 = False
        for i, v_ in enumerate(diameter_path):
            v = g.vs[v_]
            if i == 0:
                if v['type'] != 8:
                    cond2 = False
            elif i == len(diameter_path) - 1:
                if v['type'] != 9:
                    cond2 = False
            else:
                #if v['type'] not in [1,2,3]: # main path nodes must come from subg_type = 6 or 7 or 10 or 11
                if v['type'] in [4, 5]:
                    cond2 = False
                    predecessors_ = g.predecessors(i)
                    successors_ = g.successors(i)
                    for v_p in predecessors_:
                        v_p_succ = g.successors(v_p)
                        for v_cand in v_p_succ:
                            inster_set = set(g.successors(v_cand)) & set(successors_)
                            if g.vs[v_cand]['type'] in [0,1] and len(inster_set) > 0:
                                cond2 = True
        return cond1 and cond2


def extract_latent_z(data, model, data_type='igraph', start_idx=0, infer_batch_size=64, device=None):
    model.eval()
    Z = []
    g_batch = []
    for i, g  in enumerate(tqdm(data)):
        if data_type== 'tensor':
            g_ = g.to(device)
        elif data_type== 'pygraph':
            g_ = deepcopy(g)
        else:
            g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == infer_batch_size or i == len(data) - 1:

            g_batch = model._collate_fn(g_batch)
            if data_type == 'pygraph':
                g_batch = DataBatch.from_data_list(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
    
    return np.concatenate(Z, 0)


def prior_validity(train_data, model, infer_batch_size=64, data_type='igraph', subg=True, device=None, scale_to_train_range=False):
    # data_type: igraph, pygraph
    
    if scale_to_train_range:
        Z_train = extract_latent_z(train_data, model, data_type, 0, infer_batch_size, device=device)
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    
    n_latent_points = 1000
    decode_times = 10
    valid_dags = 0
    valid_ckts = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    Ckt_valid = []
    if data_type == 'igraph':
        G_train = train_data
    elif data_type == 'pygraph':
        G_train = [pygraph_to_igraph(g) for g in train_data]
    else:
        raise NotImplementedError()
    
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == infer_batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model.get_device())
            if scale_to_train_range:
                z = z * z_std + z_mean  # move to train's latent range
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                for g in g_batch:
                    if is_valid_DAG(g, subg):
                        valid_dags += 1
                        G_valid.append(g)
                    if is_valid_Circuit(g, subg=subg):
                        valid_ckts += 1
                        Ckt_valid.append(g)
            cnt = 0

    r_valid_dag = valid_dags / (n_latent_points * decode_times)
    print('Ratio of valid DAG decodings from the prior: {:.4f}'.format(r_valid_dag))

    r_valid_ckt = valid_ckts / (n_latent_points * decode_times)
    print('Ratio of valid Circuits decodings from the prior: {:.4f}'.format(r_valid_ckt))

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    return r_valid_dag, r_valid_ckt, r_novel