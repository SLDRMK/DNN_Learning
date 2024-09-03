import torch.nn as nn
from .ResNet import ResNet34

# Default Neural Network use ResNet as an example
# as paper at https://arxiv.org/abs/1512.03385
class DefaultNeuralNetwork(nn.Module):
    def __init__(self):
        super(DefaultNeuralNetwork, self).__init__()
        self.resnet = ResNet34()

    def forward(self, x):
        x = self.resnet(x)
        return x