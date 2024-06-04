import torch
from siamese_network import SiameseNetwork
from triplet_loss import TripletLoss

embedding_size = 128

siamese_net = SiameseNetwork(embedding_size)
loss = TripletLoss()

