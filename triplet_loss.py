import torch.nn as nn
import torch
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance_metric = 'euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def calc_cosine_similarity(self, x1, x2):
        cos = nn.CosineSimilarity(dim = 1, eps= 1e-6)
        return cos(x1, x2)
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == 'euclidean':
            distance_positive = self.calc_euclidean(anchor, positive)
            distance_negative = self.calc_euclidean(anchor, negative)
        elif self.distance_metric == 'cosine':
            distance_positive = self.calc_cosine_similarity(anchor, positive)
            distance_negative = self.calc_cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()