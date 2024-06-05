import os
import random
from sklearn.model_selection import train_test_split

folders = []
data_dir = "D:\Datasets\Face Recognition\TestCheck"
for folder in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, folder)):
        folders.append(os.path.join(folder))
# print(folders)

# Splitting the folders into train, validation, and test sets
train_folders, test_folders = train_test_split(folders, test_size=0.1, random_state=42)
train_folders, val_folders = train_test_split(train_folders, test_size=0.25, random_state=42)

print(f'Train folders: {len(train_folders)}')
print(f'Validation folders: {len(val_folders)}')
print(f'Test folders: {len(test_folders)}')