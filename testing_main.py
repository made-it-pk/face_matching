import torch
import torch.optim as optim
from siamese_network import SiameseNetwork
from torchvision import transforms
from triplet_loss import TripletLoss
from config import EMBEDDING_SIZE, NUM_EPOCHS, LEARNING_RATE
from tqdm import tqdm
from PIL import Image
from dataset import TripletDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SiameseNetwork(EMBEDDING_SIZE).to(device)
criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

anchor_path = "D:/Datasets/Face Recognition/TestCheck/12/0.jpg"
positive_path = "D:/Datasets/Face Recognition/TestCheck/12/1.jpg"
negative_path = "D:/Datasets/Face Recognition/TestCheck/12/2.jpg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

anchor_img = transform(Image.open(anchor_path).convert('RGB')).unsqueeze(0).to(device)
positive_img = transform(Image.open(positive_path).convert('RGB')).unsqueeze(0).to(device)
negative_img = transform(Image.open(negative_path).convert('RGB')).unsqueeze(0).to(device)

for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    optimizer.zero_grad()
    
    embedding_anchor, embedding_positive, embedding_negative = model(anchor_img, positive_img, negative_img)
    
    loss = criterion(embedding_anchor, embedding_positive, embedding_negative)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'siamese_network.pth')
