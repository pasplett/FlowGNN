import numpy as np
import torch

from scipy.stats import pearsonr

def train_pygraph_supervised(model, loader, loss_function, optimizer):
    model.train()
    device = model.get_device()

    predictions, targets = [], []

    for g, y in loader:  # Iterate in batches over the training dataset.
        
        g, y = g.to(device), y.to(device)
        yhat = model.predict(g)

        yhat, y = torch.squeeze(yhat), torch.squeeze(y)
        with torch.no_grad():
            predictions.append(yhat)
            targets.append(y)
        loss = loss_function(yhat, y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    
    predictions = torch.concat(predictions, dim=0)
    targets = torch.concat(targets, dim=0)

    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    rmse = np.sqrt(np.mean((predictions - targets)**2)).item()
    pearson = float(pearsonr(predictions, targets)[0])

    return rmse, pearson


def train_igraph_supervised(
        model, dataset, loss_function, optimizer, batch_size=64
    ):
    model.train()
    device = model.get_device()

    predictions, targets = [], []

    n_data = len(dataset)
    draws = np.arange(n_data, dtype=int)
    np.random.shuffle(draws)
    
    g_batch, y_batch = [], []
    for i, sample_idx in enumerate(draws):  # Iterate in batches over the training dataset.
        
        g, y = dataset.__getitem__(sample_idx)
        g_batch.append(g)
        y_batch.append(y)
        
        if len(g_batch) == batch_size or i == n_data - 1:
        
            g_batch = model._collate_fn(g_batch)
            yhat = model.predict(g_batch)
            
            y_batch = torch.Tensor(y_batch).to(device)
            yhat, y_batch = torch.squeeze(yhat), torch.squeeze(y_batch)
            with torch.no_grad():
                predictions.append(yhat)
                targets.append(y_batch)
            loss = loss_function(yhat, y_batch)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            
            g_batch, y_batch = [], []
    
    predictions = torch.concat(predictions, dim=0)
    targets = torch.concat(targets, dim=0)

    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    rmse = np.sqrt(np.mean((predictions - targets)**2)).item()
    pearson = float(pearsonr(predictions, targets)[0])

    return rmse, pearson


def test_pygraph_supervised(model, loader, return_predictions=False):
    model.eval()
    device = model.get_device()

    predictions, targets = [], []
    
    with torch.no_grad():
        for g, y in loader:
            
            g, y = g.to(device), y.to(device)
            yhat = model.predict(g)
            
            yhat, y = torch.squeeze(yhat), torch.squeeze(y)
            predictions.append(yhat)
            targets.append(y)
    
    predictions = torch.concat(predictions, dim=0)
    targets = torch.concat(targets, dim=0)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    rmse = np.sqrt(np.mean((predictions - targets)**2)).item()
    pearson = float(pearsonr(predictions, targets)[0])

    if return_predictions:
        return predictions, targets, rmse, pearson
    else:
        return rmse, pearson


def test_igraph_supervised(model, dataset, batch_size=64, return_predictions=False):
    model.eval()
    device = model.get_device()
    
    predictions, targets = [], []
        
    n_data = len(dataset)
    draws = np.arange(n_data, dtype=int)
    np.random.shuffle(draws)
    
    with torch.no_grad():
        g_batch, y_batch = [], []
        for i, sample_idx in enumerate(draws):  # Iterate in batches over the test dataset.
            
            g, y = dataset.__getitem__(sample_idx)
            g_batch.append(g)
            y_batch.append(y)
        
            if len(g_batch) == batch_size or i == n_data - 1:
                
                g_batch = model._collate_fn(g_batch)
                yhat = model.predict(g_batch)
                
                y_batch = torch.Tensor(y_batch).to(device)
                yhat, y_batch = torch.squeeze(yhat), torch.squeeze(y_batch)
                predictions.append(yhat)
                targets.append(y_batch)
                
                g_batch, y_batch = [], []
    
    predictions = torch.concat(predictions, dim=0)
    targets = torch.concat(targets, dim=0)

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    rmse = np.sqrt(np.mean((predictions - targets)**2)).item()
    pearson = float(pearsonr(predictions, targets)[0])

    if return_predictions:
        return predictions, targets, rmse, pearson
    else:
        return rmse, pearson