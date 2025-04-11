# New task: how much should we train the model? how many iterations? effect on clusters?
import os 
import sys
import gc
import time
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import logging
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint_sequential
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import wandb
try: from hflayers import Hopfield
except ModuleNotFoundError: pass
from sklearn.model_selection import train_test_split

from CSIP.utils import unload_data, analyze_G, encode
from CSIP.dataset import GraphDataset, GraphLoader
from CSIP.model.GIN import GNN
from CSIP.training import train, evaluate


def get_cosine_scheduler(
        optimizer: Optimizer, warmup: int = 5, 
        num_training_steps: int = 20, num_cycles: float = 0.5, last_epoch: int = -1 
):
    '''
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        warmup (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    '''

    def lr_lambda(step):
        if step < warmup:

            # return float(step) / float(max(1, warmup)) # more conservative
            return 1.0 # aggressive
        
        progress = float(step - warmup) / float(max(1, num_training_steps - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# tunable hyperparameters: [lr, batch_size, hidden_dim, 
            # loss, num_mlps, num_layers, pad, graph_dim, dropout]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type = str, default = None, help = "Path to dataset")
    parser.add_argument("--load_dir_label", type = str, default = None, help = "Path to labels set")
    parser.add_argument("--save_dir", type = str, default = 'output', help = "Where to save output files.")
    parser.add_argument("--device", type = str, default = "cuda", help = "device")
    parser.add_argument("--loss", type = str, default = "standard", help = "type of loss function, can be either 'standard' or 'cloob' ")
    parser.add_argument("--lr", type = float, default = 0.01, help = "initial learning rate")
    parser.add_argument("--batch_size", type = int, default = 4096, help = "batch size")
    parser.add_argument("--dropout", type = float, default = 0.3, help = "proportion of edges to dropout in each graph")
    parser.add_argument("--iters", type = int, default = 100, help = "number of epochs to train for")
    parser.add_argument("--eval_batch", type = int, default = 1, help = "number of batches to use during evaluation" )
    parser.add_argument("--train_size", type = float, default = 0.85, help = "proportion of data to be used for training")
    parser.add_argument("--ckpt", type = float, default = None, help = "path to load parameters from")
    parser.add_argument("--input_dim", type = int, default = 256, help = "size of input node feature")
    parser.add_argument("--hidden_dim", type = int, default = 64, help = "size of node representation")
    parser.add_argument("--num_mlps", type = int, default = 2, help = "number of linear layers to use after each node aggregation")
    parser.add_argument("--num_layers", type = int, default = 3, help = "number of layers")
    parser.add_argument("--pad", type = int, default = 80, help = "how many nodes to use in each graph (all graphs are padded to have the same number of nodes) ")
    parser.add_argument("--graph_dim", type = int, default = 64, help = "size of graph representation")
    parser.add_argument("--scale_hopfield", type = int, default = None, help = "")
    parser.add_argument("--learnable_tau", type = int, default = 1, help = "whether we learn tau")
    parser.add_argument("--precision", type = str, default = "amp", help = "whether we use mixed precision to speed up training. Can be either 'amp' or 'none' ")
    parser.add_argument("--use_tensorboard", type = int, default = 0, help = "")
    parser.add_argument("--use_wandb", type = int, default = 0, help = "")
    parser.add_argument("--debug", type = int, default = 0, help = "only set to true if you use wandb")

    args = parser.parse_args()

    # model 
    model_img = GNN(args.input_dim, num_layers = args.num_layers,
                hidden_dim = args.hidden_dim, num_mlps = args.num_mlps, 
                pad = args.pad, graph_dim=args.graph_dim, 
                learnable_inv_tau = args.learnable_tau)

    model_mol = GNN(45, num_layers = args.num_layers,
                hidden_dim = args.hidden_dim, num_mlps = args.num_mlps, 
                pad = 45, graph_dim=args.graph_dim,
                learnable_inv_tau = args.learnable_tau)
    
    model_img.train()
    model_img = model_img.to(args.device)
    model_mol.train()
    model_mol = model_mol.to(args.device)

    # optim
    optimizer_img = optim.AdamW(model_img.parameters(), args.lr) # weight decay?
    optimizer_mol = optim.AdamW(model_mol.parameters(), args.lr)

    # if we use amp precision
    if args.precision == 'amp':
        scaler = GradScaler()

    # scheduler
    scheduler_img = get_cosine_scheduler(optimizer_img)
    scheduler_mol = get_cosine_scheduler(optimizer_mol)

    # load data
    # graph, nodes, labels = unload_data(args.load_dir, load_label=1)
    # labels = np.array(labels)

    # for i in range(len(nodes)):
    #     nodes[i] = nodes[i][0]
    # for i in range(len(graph)):
    #     graph[i] = graph[i][0]

    # nodes = np.array(nodes, dtype = object)
    # graph = np.array(graph, dtype = object)

    labels = unload_data(args.load_dir_label, load_graph = False, load_node = False, load_label = 1)[0]

    # generate dataset
    # idxs = np.argsort(labels) 
    # graph, nodes, labels = graph[idxs], nodes[idxs], labels[idxs]

    labels = np.sort(labels)
    class_begins, class_size = [0], [0]
    num_classes = 1

    for i in range(1, len(labels)):

        if labels[i] != labels[i - 1]:
            class_begins.append(i)
            class_size.append(0)
            num_classes += 1
        class_size[len(class_size) - 1] += 1

    class_begins.append(len(labels))
    classes = np.arange(num_classes)
    np.random.shuffle(classes)

    # zero-shot
    # train_classes = round(num_classes * args.train_size)    
    # train_idxs = np.array([j for i in classes[:train_classes] for j in range(class_begins[i], class_begins[i + 1])] )
    # test_idxs = np.array([j for i in classes[train_classes:] for j in range(class_begins[i], class_begins[i + 1])] )

    # stratified on most frequent class
    train_idxs, test_idxs = [], []
    for i in range(len(class_size)):

        if class_size[i] > 1000: continue ########## skip DMSO, needed?

        if class_size[i] < 18: 
            train_idxs.extend([i for i in range(class_begins[i], class_begins[i + 1])])
            continue
        n_samples = int(args.train_size * class_size[i])
        idxs = np.random.permutation(range(class_begins[i], class_begins[i + 1]))
        train_idxs.extend(idxs[:n_samples])
        test_idxs.extend(idxs[n_samples:])

    np.random.shuffle(train_idxs)
    np.random.shuffle(test_idxs)
    np.random.shuffle(test_idxs)

    train_dataset = GraphDataset(labels = labels, path = args.load_dir, idxs = train_idxs, 
                                 pad = args.pad, size = args.input_dim, 
                                 drop_edge=args.dropout, norm = False) 
    stats, stats_label = train_dataset.stats, train_dataset.stats_label
    train_loader = GraphLoader(train_dataset, batch_size = args.batch_size)
    test_dataset = GraphDataset(labels = labels, path = args.load_dir, idxs = test_idxs, 
                                pad = args.pad, size = args.input_dim, drop_edge=0.0,
                                stats = stats, stats_label = stats_label)
    test_loader = GraphLoader(test_dataset, batch_size = args.batch_size) # check
    args.batch_per_epoch = max(1, int(len(train_loader) / args.batch_size))

    # hopfield
    if args.loss == 'cloob':
        hopfield_layer = Hopfield(input_size=args.hidden_dim,
                                scaling=args.scale_hopfield,
                                normalize_hopfield_space=False,
                                normalize_hopfield_space_affine=False,
                                normalize_pattern_projection=False,
                                normalize_pattern_projection_affine=False,
                                normalize_state_pattern=False,
                                normalize_state_pattern_affine=False,
                                normalize_stored_pattern=False,
                                normalize_stored_pattern_affine=False,
                                state_pattern_as_static=True,
                                pattern_projection_as_static=True,
                                stored_pattern_as_static=True,
                                disable_out_projection=True,
                                num_heads = 1,
                                dropout=False)

    # training
    gc.collect()
    model_img.train()
    model_mol.train()
    logging.getLogger().setLevel(logging.INFO)
    Writer = None
    
    if args.ckpt is not None:

        ckpt = torch.load(args.ckpt)
        start_iter = ckpt['iter']

        model_img.load_state_dict(ckpt['model_img_state'])
        optimizer_img.load_state_dict(ckpt['optimizer_img'])
        if scheduler_img is not None and "scheduler_img" in ckpt:
            scheduler_img.load_state_dict(ckpt['scheduler_img'])

        model_mol.load_state_dict(ckpt['model_mol_state'])
        optimizer_mol.load_state_dict(ckpt['optimizer_mol'])
        if scheduler_mol is not None and "scheduler_mol" in ckpt:
            scheduler_mol.load_state_dict(ckpt['scheduler_mol'])

        logging.info("All keys are matched successfully.")
        
    else:
        logging.info("Checkpoint not available. Using random initialization instead.")

    if args.use_tensorboard == True:
        Writer = SummaryWriter(args.save_dir)

    if args.use_wandb == True:
        logging.debug("Starting wandb.")
        wandb.init(
            project = 'img2mol'
        )
        if args.debug:
            wandb.watch(model_img, log = 'all')
            wandb.watch(model_mol, log = 'all')

        logging.debug("Finish loading wandb.")

    iters_per_epoch = int(len(train_loader) / args.batch_size)
    scheduler_img.step()
    scheduler_mol.step()

    args.eval_step = 3

    for i in tqdm(range(start_iter, args.iters * args.batch_per_epoch)):

        if (i + 1) % (args.eval_step) == 0: #((i + 1) % iters_per_epoch == 0) & (i >= 1):

            scheduler_img.step()
            scheduler_mol.step()

            # for j in range(args.eval_batch):

            #     features, graphs, labels, number_of_nodes = next(test_loader)
            #     features = features.to(args.device)
            #     labels = labels.to(args.device)
            #     with torch.no_grad():
            #         preds = model(graphs, number_of_nodes, features)
            #     preds = np.argmax(preds, axis = 1)
            #     labels = np.argmax(labels, axis = 1)
            #     eval_acc += sum(preds == labels)

            # eval_acc = eval_acc.item() / (args.eval_batch * args.batch_size)

            # for j in range(args.eval_batch):

            #     features, graphs, labels, number_of_nodes = next(train_loader)
            #     features = features.to(args.device)
            #     labels = labels.to(args.device)

            #     with torch.no_grad():
            #         preds = model(graphs, number_of_nodes, features)
            #     preds = np.argmax(preds, axis = 1)
            #     labels = np.argmax(labels, axis = 1)
            #     train_acc += sum(preds == labels)
            
            # train_acc = train_acc.item() / (args.eval_batch * args.batch_size)
            # logging.info(f'eval on iters {i + 1}, eval_acc = {eval_acc:3f}, train_acc = {train_acc:3f}')
            
            # model_img.train()
            # model_mol.train()
            
            logging.info(f"evaluating...")

            logging.info(f"--------------- Test data Eval ---------------")
            for _ in range(args.eval_batch):

                features, graphs, (labels_graphs, labels_features), _ = next(test_loader)
                features = features.to(args.device)

                batch = ((graphs, features), (labels_graphs, labels_features))
                    
                evaluate(model_img, model_mol, batch, args, 
                    n_iter = i, tb_writer=None)#Writer)
                
            logging.info(f"--------------- Train data eval ---------------")

            for _ in range(args.eval_batch):

                features, graphs, (labels_graphs, labels_features), _ = next(train_loader)
                features = features.to(args.device)

                batch = ((graphs, features), (labels_graphs, labels_features))
                    
                evaluate(model_img, model_mol, batch, args, 
                    n_iter = i, zero_shot = False, tb_writer=None)#Writer)

            logging.info(f"--------------- Eval complete ---------------")


        if (i + 1) % (args.batch_per_epoch) == 0:
            logging.info(f"iters {i + 1}, saving...")
            state = {
                "iter": i + 1,
                "model_img_state": model_img.state_dict(),
                "model_mol_state": model_mol.state_dict(),
                "optimizer_img": optimizer_img.state_dict(),
                "optimizer_mol": optimizer_mol.state_dict(),
                "scheduler_img": scheduler_img.state_dict(),
                "scheduler_mol": scheduler_mol.state_dict()
            }
            # torch.save(state, os.path.join(args.save_dir, f'{i+1}.pth'))
        
        features, graphs, (labels_graphs, labels_features), _ = next(train_loader)
        features = features.to(args.device)

        batch = ((graphs, features), (labels_graphs, labels_features))
        
        train(model_img, model_mol, optimizer_img, optimizer_mol, 
            scaler, batch, args, 
            n_iter = i, tb_writer=None)#Writer)
        
        # optimizer.zero_grad()

        # with torch.cpu.amp.autocast(): # change to cuda if using cuda
        
        # # with torch.autograd.set_detect_anomaly(True):
        #     preds = model(graphs, number_of_nodes, features) # graph_feature = torch.sum(node_features, dim = 1) ?
        #     loss = criterion(preds, labels) # labels assignment not correct? nodes not enough?
        #     loss.backward()
        #     optimizer.step()

    logging.debug("All done!")

        
