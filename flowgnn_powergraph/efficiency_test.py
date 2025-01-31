"""
File to train the GNN model.
"""

import os
import numpy as np
import torch
import warnings
from torch.optim import Adam
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_data_args,
    get_graph_size_args,
)
from utils.io_utils import check_dir
from gendata import get_dataloader, get_dataset
import torch.nn.functional as F
from gnn.model import get_gnnNets
from utils.path import MODEL_DIR
import time

# Save directory model_name + dataset_name + layers + hidden_dim

class TrainModel(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        seed,
        graph_classification=False,
        graph_regression=True,
        save_dir=None,
        save_name="model",
        **kwargs,
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.loader = None
        self.device = device
        self.graph_classification = graph_classification
        self.graph_regression = graph_regression
        #self.node_classification = not graph_classification
        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        self.seed = seed
        check_dir(self.save_dir)

        if self.graph_classification or self.graph_regression:
            self.dataloader_params = kwargs.get("dataloader_params")
            self.loader = get_dataloader(dataset, **self.dataloader_params)

    def __loss__(self, logits, labels):
        if self.graph_classification:
            return F.nll_loss(logits, labels)
        elif self.graph_regression:
            return F.mse_loss(logits.squeeze(), labels)

    # Get the loss, apply optimizer, backprop and return the loss

    def _train_batch(self, data, labels):
        logits = self.model(data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
            #print(loss)
        elif self.graph_regression:
            loss = self.__loss__(logits, labels)
        else:
            loss = self.__loss__(logits[data.train_mask], labels[data.train_mask])           

        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        logits = self.model(data)
        if self.graph_classification:
            loss = self.__loss__(logits, labels)
            loss = loss.item()
            preds = logits.argmax(-1)
            return loss, preds, logits
        elif self.graph_regression:
            loss = self.__loss__(logits, labels)
            loss = loss.item()
            preds = logits
            return loss, preds
        else:
            mask = kwargs.get("mask")
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(logits[mask], labels[mask])
            loss = loss.item()
            preds = logits.argmax(-1)

        return loss, preds

    def test(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            for batch in self.loader[0]["train"]:
                batch = batch.to(self.device)
                _, __, ___ = self._eval_batch(batch, batch.y)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        print(
            f"Inference Time: {np.mean(times)} +- {np.std(times)} s"
        )
        return

    # Train model
    def train(self, train_params=None, optimizer_params=None):
        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)
        self.model.to(self.device)
        self.model.train()

        times = []
        for i in range(10):
            t0 = time.perf_counter()
            for batch in self.loader[0]["train"]:
                batch = batch.to(self.device)
                _ = self._train_batch(batch, batch.y)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        print(
            f"Training Time: {np.mean(times)} +- {np.std(times)} s"
        )
        return

#  Main train function
def train_gnn(args, args_group):
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dev {device}")

    dataset_params = args_group["dataset_params"]
    model_params = args_group["model_params"]
    model_params["seed"] = args.seed
    if args.datatype == 'regression':
        model_params['graph_regression'] = "True"
    else:
        model_params['graph_regression'] = "False"

    # changing the dataset path here, load the dataset
    dataset = get_dataset(
        dataset_root=os.path.join(args.data_save_dir, args.dataset_name),
        **dataset_params,
    )
    dataset.data.x = dataset.data.x.float()

    if args.datatype == 'regression':
        args.graph_regression = "True"
        args.graph_classification = "False"
        dataset.data.y = dataset.data.y.squeeze().float()

    else:
        args.graph_regression = "False"
        args.graph_classification = "True"
        dataset.data.y = dataset.data.y.squeeze().long()
    # get dataset args
    args = get_data_args(dataset, args)


    if eval(args.graph_classification) | eval(args.graph_regression):
        dataloader_params = {
            "batch_size": args.batch_size,
            "random_split_flag": eval(args.random_split_flag),
            "data_split_ratio": [args.train_ratio, args.val_ratio, args.test_ratio],
            "seed": args.seed,
        }
    # get model
    model = get_gnnNets(args.num_node_features, args.num_classes, model_params, eval(args.graph_regression))

    # train model
    if eval(args.graph_classification):
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            seed=args.seed,
            graph_classification=eval(args.graph_classification),
            graph_regression=eval(args.graph_regression),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=f"{args.dataset_name}_{args.model_name}_{args.datatype}_{args.num_layers}l_{args.hidden_dim}hid_{args.heads}h_{args.seed}s",
            dataloader_params=dataloader_params,
        )
    else:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            seed=args.seed,
            graph_classification=eval(args.graph_classification),
            graph_regression=eval(args.graph_regression),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=f"{args.dataset_name}_{args.model_name}_{args.datatype}_{args.num_layers}l_{args.hidden_dim}hid_{args.heads}h_{args.seed}s",
            dataloader_params=dataloader_params,
        ) 

    # Train model for 1 epoch
    trainer.train(
        train_params=args_group["train_params"],
        optimizer_params=args_group["optimizer_params"],
    )

    # Test model on training set
    trainer.test()


if __name__ == "__main__":
    parser, args = arg_parse()
    args = get_graph_size_args(args)

    # for loop the training architecture for the number of layers and hidden dimensions
    rnd_seeds = [0]
    tasks = ['multiclass'] #['binary', 'multiclass', 'regression']
    powergrids = ['ieee24', 'uk', 'ieee39', 'ieee118']
    models = ['gat', 'gatv2', 'gt', 'flowgat', 'flowgatv2', 'flowtransformer']
    for powergrid in powergrids:
        args.dataset_name = powergrid
        for task in tasks:
            args.datatype = task
            for num_layers in [3]:
                args.num_layers = num_layers
                for hidden_dim in [32]:
                    args.hidden_dim = hidden_dim
                    for model in models:
                        args.model_name = model
                        for rnd_seed in rnd_seeds:
                            args.seed = rnd_seed

                            if os.path.exists(MODEL_DIR + f"{powergrid}/{powergrid}_{model}_{task}_{num_layers}l_{args.hidden_dim}hid_{rnd_seed}s_scores.json"):
                                continue

                            fix_random_seed(rnd_seed)
                            args_group = create_args_group(parser, args)
                            print(f"Hidden_dim: {args.hidden_dim}, Num_layers: {args.num_layers}, model {args.model_name}, data {args.dataset_name}, task {args.datatype}, rnd_seed {rnd_seed} ")
                            train_gnn(args, args_group)

        print("CHANGE POWERGRID")

    print("END TRAINING OF POWERGRAPH")
