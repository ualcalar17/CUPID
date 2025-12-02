import torch.nn as nn
from models.resnet import ResNet
from src.models.data_consistency import Data_consistency

class UnrolledNet(nn.Module):
    def __init__(self, device, **kwargs):
        super(UnrolledNet, self).__init__()

        self.nb_unroll_blocks = kwargs.get("nb_unroll_blocks")
        resnet_kwargs = kwargs.get("ResNet")
        dc_kwargs = kwargs.get("DC")
        self.resnet = ResNet(device, **resnet_kwargs)
        self.dc = Data_consistency(**dc_kwargs)
        
    def forward(self, zerofilled, coil, mask):

        output = zerofilled.clone()
        
        for _ in range(self.nb_unroll_blocks):
        
            output = self.resnet(output)
            output, mu = self.dc(zerofilled, coil, mask, output)
        
        return output, mu