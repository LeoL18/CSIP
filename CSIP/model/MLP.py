import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNorm(nn.Module):

    def __init__(self, hidden_dim):
        
        super(GraphNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.alpha = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
    
    def forward(self, x):

        if x.dim() not in [2, 3]: raise ValueError("x must have dim in [2, 3].")

        if x.dim() == 2:
            return self.norm(x)

        x_norm = torch.zeros_like(x)
        for i in range(x.size(0)):
            x_norm[i] = self.norm(x[i])
            
        return x_norm

class MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        '''
        num_layers: the number of layers in the network excluding the input layer. 
        input_dim: dimensionality of the input features
        hidden_dim: dimensionality of the hidden features at all non-input layers
        output_dim: dimensionally of the final output
        '''

        super(MLP, self).__init__()

        self.num_layers = num_layers

        if num_layers < 1: raise ValueError('The number of layers must be greater than 0')
        elif num_layers == 1: 
            self.final_linear = nn.Linear(input_dim, output_dim)
        else: 

            self.layers = torch.nn.ModuleList()

            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.GELU())
            self.layers.append(GraphNorm(hidden_dim))

            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.GELU()) # better performance?
                self.layers.append(GraphNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(GraphNorm(hidden_dim))

            self.final_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.num_layers == 1: return self.final_linear(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return self.final_linear(x)
            
