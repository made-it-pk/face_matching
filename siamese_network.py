import torch
import torch.nn as nn
from torchvision.models import resnet18

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=128):
        super(SiameseNetwork, self).__init__()
        self.resnet = resnet18(weights = 'ResNet18_Weights.DEFAULT')
        self.embedding_size = embedding_size
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)
        
    def forward(self, x1, x2, x3):
        output1 = self.resnet(x1)
        output2 = self.resnet(x2)
        output3 = self.resnet(x3)
        return output1, output2, output3
