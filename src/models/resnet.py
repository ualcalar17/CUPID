import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Residual_Block(nn.Module):
    def __init__(self, device, channels, kernel_size, stride, padding, bias, scalar):
        super(Residual_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias).to(device)
        self.relu = nn.ReLU(inplace=True)
        self.scalar = nn.Parameter(torch.tensor(scalar), requires_grad=False).to(device)

    def forward(self, inp):
        x = self.conv(inp)
        x = self.relu(x)        
        x = self.conv(x)
        x = self.scalar*x
        return x + inp

class ResNet(nn.Module):
    def __init__(self, device, nb_res_blocks, channels, kernel_size, stride, padding, bias, scalar, use_checkpointing, weights_mean, weights_std):
        super(ResNet, self).__init__()
        self.first_layer = nn.Conv2d(in_channels = 2, out_channels = channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
        res_block = []
        for _ in range(nb_res_blocks):
            res_block += [Residual_Block(device, channels, kernel_size, stride, padding, bias, scalar)]
            
        self.res_block = nn.Sequential(*res_block)

        self.last_layer = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.final_conv = nn.Conv2d(in_channels = channels, out_channels = 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.initialize_weights(weights_mean, weights_std)
        self.use_checkpointing = use_checkpointing

    def initialize_weights(self, weights_mean, weights_std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=weights_mean, std=weights_std)

    def forward(self, input_data):
        z = self.first_layer(input_data)
        if self.use_checkpointing:
            output = checkpoint(self.res_block, z, use_reentrant=True)
        else:
            output = self.res_block(z)
        output = self.last_layer(output)
        output = output + z
        output = self.final_conv(output)
        return output