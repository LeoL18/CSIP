import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
import os

import torch
from torch.utils.data import Dataset

from .preprocessing import process_molecule

def normalize(data, mean = None, std = None):

    if (mean is None) | (std is None):
        mean = torch.zeros_like(data[0][0])
        std = torch.zeros_like(data[0][0])
        size = 0.

        # calculate the mean vector
        for i in range(len(data)):
            for j in range(len(data[i])):
                mean += data[i][j]
            size += len(data[i])

        mean /= size

        # calculate variance
        for i in range(len(data)):
            for j in range(len(data[i])):
                std += (data[i][j] - mean) ** 2
        
        std = torch.sqrt(std / size)

    eps = 1e-9
    
    # normalize
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - mean) / (std + eps)

    return (mean, std)

def clamp_(features, clamp_size):
    
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j] =  torch.clamp(features[i][j], min = -clamp_size, max = clamp_size) 

class GraphDataset(Dataset):

    def __init__(self, features = None, graphs = None, labels = None, path = None, 
                 pad = 80, size = 256, drop_edge = 0.3, clamp = 4.0, idxs = None,
                 index = None, stats = None, stats_label = None, norm = False):
        
        '''
        Base dataset for model training

        Args:

        features: node features, expect 3d input, not necessarily having uniform size across dimension
        graphs: a list of networkx objects
        labels: a list of SMILES string
        path: for large dataset, the dataset can instead use path to prevent RAM overflow.  
        Expect input to be shaped like
        Dataset
            feature_000001
            feature_000002
            ...
            graph_000001
            graph_000002
            ...
        pad: how many nodes to use for each graph 
        size: size of input feature vector
        drop_edge: proportion of edges to drop during training
        clamp: maximum node feature value, used to prevent slow convergence due to large numbers
        stats: if used, the dataset normalize each node feature accordingly. Expect input like (mean, std)
        stats_label: normalizing statistics for graphs generated from SMILES code. Expect input like (mean, std)

        Returns:
        base dataset with processed data
        '''
        
        self.use_path = self.stats = False        
        self.point = 0
        self.index = self.idxs = None
        self.drop_edge = drop_edge
        self.clamp = clamp

        if idxs is not None:    
            self.idxs = idxs

        if index is not None:
            self.index = {}
            for i in range(len(index)):
                self.index[index[i]] = i

        if path is not None: 
            self.use_path = True
            self.paths = np.array([(path_1, path_2) for (path_1, path_2) in zip(sorted(glob(os.path.join(path, 'graph_*'))), 
                                   sorted(glob(os.path.join(path, 'feature_*'))))])

            if self.idxs is not None: self.paths = self.paths[self.idxs]

        elif (features is not None) & (graphs is not None):
            self.features = features
            self.graphs = graphs
            if self.idxs is not None:
                self.features = features[self.idxs]
                self.graphs = graphs[self.idxs]

        else: 
            raise ValueError('No data is provided')
        
        if stats is not None:
            self.stats = stats
        
        if labels is not None:

            self.labels_graph, self.labels_features = [], []

            for i in tqdm(range(len(labels)), desc = 'Processing SMILES...'):
                graph, feature = process_molecule(labels[i])
                self.labels_graph.append(graph)
                self.labels_features.append(feature)

            if stats_label != None:
                normalize(self.labels_features, *stats_label)
            else:
                self.stats_label = normalize(self.labels_features)

            self.pad = self.input_dims = 45
            self.labels_features, self.labels_graphs, _ = self.pad_cell(self.labels_graph, self.labels_features)

        else: self.labels_graph, self.labels_features = None, None

        self.pad = pad
        self.input_dims = size

        if self.use_path:
            
            mean = torch.zeros(256)
            std = torch.ones(256)
            self.stats = (mean, std)
            tot = 0.

            if norm:  
                for path in self.paths:
                    with open(path[1], 'rb') as f:
                        feature = pickle.load(f)
                    mean += torch.stack(feature).sum(dim = 0)
                    tot += len(feature)
                
                mean /= tot
                
                for path in self.paths:
                    with open(path[1], 'rb') as f:
                        feature = pickle.load(f)
                    feature = torch.stack(feature)
                    std += ((feature - mean) ** 2).sum(dim = 0)
                
                self.stats = (mean, np.sqrt(std / tot))
            return

        if self.stats: normalize(self.features, *stats)
        else: self.stats = normalize(self.features)

        self.features, self.graphs, self.number_of_nodes = self.pad_cell(self.graphs, self.features)

        # clamp to prevent slow training due to large values
        clamp_(self.features, self.clamp)
        return 

        
    def get_graph(self, path):

        with open(path, "rb") as f:
            graph = pickle.load(f)

        for u, v in graph.edges():
            if np.random.randint(1, 10000 + 1)/10000 < self.drop_edge:
                graph.remove_edge(u, v)

        return graph
    
    def __len__(self):

        try:
            return len(self.paths)
        except AttributeError:
            return len(self.graphs)
    
    def pad_cell(self, G, features = None):

        # Note that this method directly writes on the memory address 
        # and does not make a deep copy
        number_of_nodes = []

        for i in range(len(G)):

            number_of_nodes.append(min(G[i].number_of_nodes(), self.pad))

            while (s := G[i].number_of_nodes()) > self.pad:
                G[i].remove_node(s)
                if features is not None:
                    features[i].pop(s - 1)

            while (s := G[i].number_of_nodes()) < self.pad:
                G[i].add_node(s + 1)
                if features is not None:
                    features[i].append(torch.zeros(self.input_dims))

            if (features is not None) & (type(features[i]) != type(torch.tensor([]))):
                features[i] = torch.stack(features[i])

        if features is not None:
            features = torch.stack([*features]).to(torch.float32)
            return features, G, np.array(number_of_nodes)
        
        else:
            return G, np.array(number_of_nodes)
    
    def __getitem__(self, i):

        if self.index is not None:
            i = self.index[i]

        if self.use_path:

            g = self.get_graph(self.paths[i][0])
            with open(self.paths[i][1], 'rb') as f:
                feature = pickle.load(f)
            normalize([feature], *self.stats)
            _, _, n = self.pad_cell([g], [feature])
            feature = torch.stack(feature)
            clamp_([feature], self.clamp)
            return (feature, g, (self.labels_graph[i], self.labels_features[i]), n)
            
        if self.labels_graph is not None:
            return (self.features[i], self.graphs[i], 
                    (self.labels_graph[i], self.labels_features[i]), self.number_of_nodes[i])
        else:
            return (self.features[i], self.graphs[i], 
                    self.number_of_nodes[i])

class GraphLoader:

    def __init__(self, dataset, batch_size):

        '''
        Base loader for model training

        dataset: GraphDataset
        batch_size: batch size
        '''

        self.dataset = dataset
        self.batch_size = batch_size
        self.cur = 0
        self.__shuffle()

        if self.batch_size > len(dataset):
            self.batch_size = len(dataset)
        
    def __shuffle(self):
        self.idxs = np.random.permutation(len(self.dataset))
    
    def __iter__(self):
        self.cur = 0
        self.__shuffle()
        return self
    
    def __len__(self):
        return len(self.dataset)

    def __next__(self):
        
        if self.cur + self.batch_size > len(self.dataset):
            self.cur = 0
            self.__shuffle()

        features, graphs, labels_graphs, labels_features, number_of_nodes = [], [], [], [], []
        
        for i in range(self.cur, self.cur + self.batch_size):

            idx = self.idxs[i]

            data = self.dataset[idx]
            features.append(data[0])
            graphs.append(data[1])

            label_graph, label_feature = data[2]
            labels_graphs.append(label_graph)
            labels_features.append(label_feature)

            number_of_nodes.append(data[3])

        self.cur += self.batch_size

        return torch.stack(features), graphs, (labels_graphs, labels_features), torch.tensor(number_of_nodes, dtype = torch.float32) # node size different

