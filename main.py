import torch
import torch.optim as optim
from siamese_network import SiameseNetwork
from triplet_loss import TripletLoss
from config import EMBEDDING_SIZE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, TRAIN_CSV_FILE_PATH
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import TripletDataset
import wandb
import importlib.util

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Please Enter the Choice for the Model to be used for training...')
print('1. resnet18')
print('2. resnet34')
print('3. efficient_net_v2_s')
print('4. Custom model')

number = int(input("Please enter the number: "))
print('---------------------')
if number == 1:
    model_name = 'resnet18'
elif number == 2:
    model_name = 'resnet34'
elif number == 3:
    model_name = 'efficient_net_v2_s'
elif number == 4:
    model_name = 'custom'
    
    custom_model_path = input("Please enter the path to the custom model script: ")
    custom_class_name = input("Please enter the name of the custom model class: ")
    
    spec = importlib.util.spec_from_file_location("custom_model", custom_model_path)
    
    custom_model = importlib.util.module_from_spec(spec)
    
    spec.loader.exec_module(custom_model)
else:
    raise ValueError(f"Unsupported model number..!!")

print('\n')
print('Please Enter the Choice of loss function to be used for Triplet loss calculation...')
print('1. Euclidean')
print('2. Cosine Similarity')
loss_type = int(input("Please enter the number: "))
print('---------------------')
if loss_type == 1:
    distance_metric = 'euclidean'
elif loss_type == 2:
    distance_metric = 'cosine'
else:
    raise ValueError(f"Unsupported loss type number..!!")


print('\n\nStarting the Training...\n\n')


# Initialize wandb
wandb.init(project='siamese_network_training', config={
    "model_name": model_name,
    "embedding_size": EMBEDDING_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "loss_type": 'Euclidean' if loss_type == 1 else 'Cosine Similarity',
    "train_csv_file_path": TRAIN_CSV_FILE_PATH
})
config = wandb.config


# Loading the model
if model_name == 'custom':
    # Dynamically get the class from the imported custom_model module
    CustomModelClass = getattr(custom_model, custom_class_name)
    model = CustomModelClass(embedding_size=EMBEDDING_SIZE).to(device)
else:
    model = SiameseNetwork(model_name=model_name, embedding_size=EMBEDDING_SIZE).to(device)

#loading the loss function class
criterion = TripletLoss(distance_metric=distance_metric)

#loading the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loading the dataset and applying transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

triplet_dataset = TripletDataset(TRAIN_CSV_FILE_PATH, transform=transform)
triplet_dataloader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    for anchor_img, positive_img, negative_img in tqdm(triplet_dataloader):
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        optimizer.zero_grad()

        embedding_anchor, embedding_positive, embedding_negative = model(anchor_img, positive_img, negative_img)

        loss = criterion(embedding_anchor, embedding_positive, embedding_negative)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(triplet_dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")
    
    # Log the average loss for this epoch to wandb
    wandb.log({"epoch": epoch+1, "loss": avg_loss})

model_save_path = 'siamese_network.pth'
# Save the model
torch.save(model.state_dict(), model_save_path)
wandb.save(model_save_path)

