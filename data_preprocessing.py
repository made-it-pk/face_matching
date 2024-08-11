import os
import random
from sklearn.model_selection import train_test_split
import csv
from config import NUM_TRIPLETS, DATA_ROOT_DIR

# Define dataset directory and CSV file paths
dataset_dir = DATA_ROOT_DIR
train_csv_file = 'train_triplets.csv'
val_csv_file = 'val_triplets.csv'
test_csv_file = 'test_triplets.csv'

# Helper function to generate triplets and then save them to a CSV file
def generate_triplets(folders, dataset_dir, csv_file):
    triplets = []
    
    for folder in folders:
        images = []
        folder_path = os.path.join(dataset_dir, folder)
        for img in os.listdir(folder_path):
            if img.endswith(('jpg', 'png', 'jpeg')):
                images.append(os.path.join(folder_path, img))
        
        num_triplets = min(NUM_TRIPLETS, len(images))
        for i in range(num_triplets):
            anchor = images[i]
            temp_negatives = []
            for _ in range(5):
                positive = random.choice([img for img in images if img != anchor])

                # Choosing a negative folder that is different from the current folder
                negative_folder = random.choice([f for f in folders if f != folder])
                
                # Prevent selecting the same negative folder multiple times
                while negative_folder in temp_negatives:
                    negative_folder = random.choice([f for f in folders if f != folder])

                temp_negatives.append(negative_folder)
                
                # Choosing a negative image randomly from the selected negative folder
                negative_images = []
                negative_folder_path = os.path.join(dataset_dir, negative_folder)
                for img in os.listdir(negative_folder_path):
                    if img.endswith(('jpg', 'png', 'jpeg')):
                        negative_images.append(os.path.join(negative_folder_path, img))
                negative = random.choice(negative_images)

                triplets.append([anchor, positive, negative])

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["anchor", "positive", "negative"])
        writer.writerows(triplets)

# Listing all folders in the dataset directory
folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Splitting the folders into train, validation, and test sets
train_folders, test_folders = train_test_split(folders, test_size=0.05, random_state=42)
train_folders, val_folders = train_test_split(train_folders, test_size=0.1, random_state=42)

print(f'Train folders: {len(train_folders)}')
print(f'Validation folders: {len(val_folders)}')
print(f'Test folders: {len(test_folders)}')

# Generating triplets for each set and saving to CSV files
generate_triplets(train_folders, dataset_dir, train_csv_file)
generate_triplets(val_folders, dataset_dir, val_csv_file)
generate_triplets(test_folders, dataset_dir, test_csv_file)
