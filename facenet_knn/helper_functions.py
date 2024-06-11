import os
from PIL import Image
import torch
import numpy as np

def load_images_from_folders(base_folder):
    person_folders = []
    for person in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person)
        if os.path.isdir(person_folder):
            person_folders.append(person_folder)
    return person_folders

def save_embedding(embedding, save_folder, image_name, label):
    filename = os.path.splitext(image_name)[0] + f'_{label}.pt'
    filepath = os.path.join(save_folder, filename)
    torch.save(embedding, filepath)
    return filepath

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_embeddings(folder_path):
    embeddings = []
    labels = []
    
    for person_folder in os.listdir(folder_path):
        person_folder_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            for file in os.listdir(person_folder_path):
                if file.endswith('.pt'):
                    embedding_path = os.path.join(person_folder_path, file)
                    embedding = torch.load(embedding_path)
                    embeddings.append(embedding.flatten())
                    labels.append(person_folder)
    return np.array(embeddings), np.array(labels)

