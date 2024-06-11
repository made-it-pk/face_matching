import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from helper_functions import load_images_from_folders, save_embedding, create_folder
from config import BASE_FOLDER, NEW_DIRECTORY

# Initialize the MTCNN module to detect faces
mtcnn = MTCNN(image_size=160, margin=0)

# Initializing the InceptionResnetV1 module to create embeddings
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        img_embedding = model(img_cropped.unsqueeze(0))
        return img_embedding.detach().numpy()
    else:
        print(f"Face not detected in image {img_path}")
        return None

def create_dataset(BASE_FOLDER, NEW_DIRECTORY):
    person_folders = load_images_from_folders(BASE_FOLDER)
    for person_folder in person_folders:
        person_name = os.path.basename(person_folder)
        print(person_name)
        new_folder = os.path.join(NEW_DIRECTORY, str(person_name))
        create_folder(new_folder)
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                embedding = get_embedding(img_path)
                if embedding is not None:
                    save_embedding(embedding, new_folder, img_name, person_name)
                    print(f"Saved embedding for {img_name} to {new_folder}")
                else:
                    print(f"Failed to generate embedding for {img_name}")


# create_dataset(BASE_FOLDER, NEW_DIRECTORY)
