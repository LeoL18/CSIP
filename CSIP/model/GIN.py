import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .MLP import MLP


class GNN(nn.Module):
    def __init__(self, input_dim, labels = 8, num_layers = 5, num_mlps = 2, 
                 hidden_dim = 256, graph_dim = 1024, pad = 160, init_inv_tau = 14.3, 
                 use_lstm = True, learn_eps = True, learnable_inv_tau = True, 
                 aggregate_neighbors = 'sum', aggregate_graph = 'sum',
                 node_feature = 'pretrained'):
        '''
        labels: the number of labels in the dataset
        num_layers: the number of layers in the network
        num_mlps: the number of mlp layers
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden features
        graph_dim: dimensionality of graph features.
        pad: how much node we use for each image. Only required when no initial embedding is provided
        Note that the input graph should be padded to the same number of nodes.
        init_inv_tau: initial value for inv_tau
        learnable_inv_tau: whether tau is learnable
        use_lstm: if we use a lstm to aggregate node features at each layer into a graph feature
        clustering: if we cluster nodes at each layer
        smoothing: if we apply smoothing
        dropout: dropout ratio in the linear layers
        node_feature: the kind of feature to use. Could be "constant" or "pretrained"
        learn_eps: when set to True, the model learns to distinguish center nodes from neighboring nodes when aggregating
        aggregate_neighbors: how to aggregate neighbor features. Can be one of 'mean', 'sum'
        aggregate_graph: how to aggregate all of the nodes in the graph. Can be one of 'mean', 'sum'
        '''

        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.aggregate_neighbors = aggregate_neighbors
        self.aggregate_graph = aggregate_graph
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.pad = pad
        self.input_dims = input_dim
        self.graph_dims = graph_dim

        # The learnable temperature parameter
        # τ was initialized to the equivalent of 0.07 from (Wu et al.,
        # 2018) and clipped to prevent scaling the logits by more
        # than 100 which we found necessary to prevent training instability.
        self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
        self.logit_inv_tau.requires_grad = bool(learnable_inv_tau)

        self.node_feature = node_feature
        if node_feature not in ['constant', 'pretrained']:
            raise ValueError("Invalid value for node_feature")
        if node_feature == 'constant':
            self.initial_feature = torch.nn.Parameter(torch.randn(hidden_dim))
            self.initial_pad = torch.nn.Parameter(torch.zeros(hidden_dim))

        self.layers = torch.nn.ModuleList()
        self.layers.append(MLP(num_mlps, input_dim, hidden_dim, hidden_dim))

        for _ in range(self.num_layers - 1):
            self.layers.append(MLP(num_mlps, hidden_dim, hidden_dim, hidden_dim))

        self.features = []
        # self.linears = torch.nn.ModuleList()
        # for _ in range(num_layers - 1):
        #    self.linears.append(nn.Linear(hidden_dim, graph_dim))
        # self.norms = torch.nn.ModuleList()
        # for _ in range(num_layers - 1):
        #    self.norms.append(nn.BatchNorm1d(graph_dim))

        if use_lstm:
            self.lstm = nn.LSTM(graph_dim, graph_dim, batch_first = True, bidirectional = True)
            self.final_linear = nn.Linear(2 * graph_dim, graph_dim)
        else: 
            # self.final_linear = nn.Linear((num_layers - 1) * graph_dim, graph_dim)
            self.lstm = None

        self.to_labels = nn.Linear(graph_dim, labels)

    def __aggregate_neighbors(self, features, graphs, layer):
        
        # (batch, *node_size, feaeture_vec_size)
        # a list of graphs
        aggregated_features = []
        for feature, graph in zip(features, graphs):
            node_features = []
            for j in range(graph.number_of_nodes()):
                try:
                    neighbor_features = torch.stack([
                        feature[n - 1] * graph[j + 1][n]['weight'] for n in graph.neighbors(j + 1)])
                    
                    new_feature = torch.sum(neighbor_features, dim = 0)

                except RuntimeError: # no neighbors
                    new_feature = torch.zeros_like(feature[j])

                except KeyError: # no edge weight 
                    neighbor_features = torch.stack([
                        feature[n - 1] for n in graph.neighbors(j + 1)]) 
                    
                    new_feature = torch.sum(neighbor_features, dim = 0)

                if self.learn_eps:
                    new_feature += (1 + self.eps[layer]) * feature[j]
                else:
                    new_feature += feature[j]
                
                if self.aggregate_neighbors == 'mean':
                    new_feature /= (len(neighbor_features) + 1)

                node_features.append(new_feature)
            aggregated_features.append(torch.stack(node_features))
        
        return torch.stack(aggregated_features)
    
    def __layer_forward(self, features, layer):
        
        #for i in range(len(features)):
            # new_features[i] = F.gelu(self.norms[layer](self.layers[layer](features[i].clone())))
        #    new_features[i] = F.gelu(self.layers[layer](features[i]))
        features = self.layers[layer](features)
        
        # graph_feature = features.sum(dim = 1) # converges to fast
        graph_feature = features.max(dim = 1)[0] # maybe?

        #----------------------------------------------------------
        # for i in range(graph_feature.shape[0]):
        #     # the diffence between the number of cells in each image is 
        #     # important to account for ????
        #     graph_feature[i] /= number_of_nodes[i] 
        #----------------------------------------------------------

        # activation after sum?
        # graph_feature = self.norms[layer](
        #     F.gelu(self.linears[layer](graph_feature)))
        self.features.append(graph_feature)

        return features
    
    def __lstm_forward(self, features): # (batch, layer - 1, graph_dims)

        prob, _ = self.lstm(features) # batch, 2 * graph_dims

        prob = F.softmax(self.final_linear(prob), dim = 1) ######
        return prob


    def __final_forward(self, get_activation): ####

        self.features = torch.stack(self.features) # (layer, batch, graph_dim)
        self.features = torch.permute(self.features, (1, 0, 2)) # (batch, layer, graph_dim)
        if self.lstm:
            prob = self.__lstm_forward(self.features)
            self.features = torch.sum(prob * self.features, dim = 1) ##########
        else: 
            self.features = torch.reshape(self.features, (self.features.shape[0], -1)) # (batch, layer * graph_dim)
            # self.features = self.final_linear(self.features)
        
        if get_activation == False:
            self.features = F.softmax(self.to_labels(self.features), dim = 1)
        return self.features

    def forward(self, graphs, features = None, number_of_nodes = None,
                get_activation = True, return_node_features = False, return_inv_tau = False):
        
        '''

        number_of_nodes: number of active nodes (non-padded nodes) in the graph. Only should be used
        when no initial node features are avaliable. 

        get_activation: returns the graph embeddings rather than the prediction scores

        return_node_features: if True, returns a tuple that additionally contains
        the node features after aggregations.

        You don't have to pass in features if you opted to use constant node embeddings
        '''

        if self.node_feature == 'constant':

            if number_of_nodes is None:
                raise ValueError("number_of_nodes is required if initial features are not available.")
            
            features = []
            for i in range(len(graphs)):
                cnt = int(number_of_nodes[i].item())
                features.append(torch.cat([self.initial_feature.repeat(
                    cnt, 1), self.initial_pad.repeat(
                        self.pad - cnt, 1)], dim = 0))

        # Graph-level features
        self.features = []

        if type(features) == type([]):
            features = torch.stack(features)

        ## + layer correct dim

        # pdb.set_trace()

        features = self.__layer_forward(features, 0)

        for i in range(self.num_layers - 1):
            
            features = self.__aggregate_neighbors(features, graphs, i) # features.sum()
            features = self.__layer_forward(features, i + 1) 

        # features = self.__layer_forward(features, i) check

            if (i == self.num_layers - 2) & return_node_features:
                node_features = features

        # pdb.set_trace()
        output = self.__final_forward(get_activation) # self.features[-1]

        if (return_node_features | return_inv_tau) == 0:
            return output
        
        returns = []
        returns.append(output)
        
        if return_node_features: 
            returns.append(node_features)
        if return_inv_tau: 
            returns.append(self.logit_inv_tau)
        
        return returns
