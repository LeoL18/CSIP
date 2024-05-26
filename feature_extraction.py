import torch 
import torch.nn as nn
import torch.nn.functional as F
    
class ConvBlockE(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 5, batchNorm = True, relu = True, pooling = True):
        super(ConvBlockE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        if batchNorm: self.norm = nn.BatchNorm2d(out_channels)
        else: self.norm = None
        if pooling: self.pool = nn.MaxPool2d(2, 2)
        else: self.pool = None
        self.relu = relu
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None: x = self.norm(x)
        if self.pool is not None: x = self.pool(x)
        if self.relu: x = F.leaky_relu(x, 0.01)
        return x
    
class ConvBlockD(nn.Module):

    def __init__(self, in_channels, out_channels, upsample = 0, kernel_size = 5, batchNorm = True, relu = True, pad = 4):
        super(ConvBlockD, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        if batchNorm: self.norm = nn.BatchNorm2d(out_channels)
        else: self.norm = None
        self.pad = pad
        self.relu = relu
        self.upsample = upsample

        with torch.no_grad():
            if self.relu: gain = ['leaky_relu', 0.01]
            else: gain = ['sigmoid']
            torch.nn.init.xavier_normal_(
                self.conv.weight.data, torch.nn.init.calculate_gain(*gain))
    
    def forward(self, x):
        if self.pad: x = F.pad(x, (self.pad,self.pad,self.pad,self.pad)) 
        x = self.conv(x)
        if self.norm is not None: x = self.norm(x)
        if self.relu: x = F.leaky_relu(x, 0.01)
        if self.upsample: x = F.interpolate(x, self.upsample, mode = 'nearest')
        return x

class ConvExtractor(nn.Module):

    def __init__(self, input_size = (68, 68, 5), channels = 32, hidden_size=256):
        super(ConvExtractor, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlockE(input_size[2], channels), # 68 -> 64 -> 32
            ConvBlockE(channels, channels),  # 32 -> 28 -> 14
            ConvBlockE(channels, channels), # 14 -> 10 -> 5
            ConvBlockE(channels, hidden_size, pooling=False,relu=False) 
        )

        self.decoder = nn.Sequential(
            ConvBlockD(hidden_size, channels, upsample = 10),
            ConvBlockD(channels, channels, upsample = 28),
            ConvBlockD(channels, channels, upsample = 64),
            ConvBlockD(channels, channels),
            ConvBlockD(channels, input_size[2], kernel_size = 1, pad = 0, relu = False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = torch.reshape(x, (x.shape[0], -1))
        # x = self.to_hidden(x)

        # x = F.dropout(x, p = 0.4, training = self.training)

        # x = self.from_hidden(x)
        # x = torch.reshape(x, (x.shape[0], 8, 8, 8))
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            x = x.flatten()
        return x

class InterpolationExtractor:
    
    def __init__(self, mode = "linear"):
        modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
        if mode not in modes: 
            raise ValueError("This mode is not supported. Mode must be one of 'nearest', 'linear', 'bilinear', 'bicubic', or 'trilinear.'")
        self.mode = mode
    
    def encode(self, x, out_dim = 8):
        interpolated = F.interpolate(x, out_dim, mode = self.mode)
        if x.dim == 3: return interpolated.flatten()
        else: return interpolated.flatten(start_dim = 1)

