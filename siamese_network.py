import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=128, model_name = 'resnet18'):
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size
        if model_name == 'resnet18':
            self.model = models.resnet18(weights = 'ResNet18_Weights.DEFAULT')
            self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights = 'ResNet34_Weights.DEFAULT')
            self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)
        elif model_name == 'efficientnetv2s':
            self.model = models.efficientnet_v2_s(weights = 'EfficientNet_V2_S_Weights')
            self.model.fc = nn.Linear(self.model.classifier[1].in_features, embedding_size)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x1, x2, x3):
        output1 = self.model(x1)
        output2 = self.model(x2)
        output3 = self.model(x3)
        return output1, output2, output3
