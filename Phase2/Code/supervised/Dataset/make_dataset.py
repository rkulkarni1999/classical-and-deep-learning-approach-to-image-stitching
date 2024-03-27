import torch
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

#######################
# Custom Dataset Class
#######################
class CocoDataset(Dataset):
    def __init__(self, path):
        self.X_data = []
        self.Y_data = []

        lst = os.listdir(path)
        for filename in lst:
            if filename.endswith("_image.npy"):
                base_filename = filename[:-10]  # Remove '_image.npy' from the filename

                # Load image and points
                image_array = np.load(os.path.join(path, base_filename + '_image.npy'))
                points_array = np.load(os.path.join(path, base_filename + '_points.npy'))

                # Preprocess and convert to tensors
                x = torch.from_numpy((image_array.astype(float) - 127.5) / 127.5).float()
                y = torch.from_numpy(points_array.astype(float) / 32.0).float()

                self.X_data.append(x)
                self.Y_data.append(y)

        self.len = len(self.X_data)

    def __getitem__(self, index):
        x, y = self.X_data[index], self.Y_data[index]
        x = x.permute(2, 0, 1)
        y = y.view(-1)
        return x, y

    def __len__(self):
        return self.len

########################################
# Function for Visualizing and Saving the Dataset Images
########################################
def visualize_and_save_dataset(dataset, index, save_path_original, save_path_warped, target_size=(128, 128)):
    # Create the directories if they don't exist
    Path(save_path_original).mkdir(parents=True, exist_ok=True)
    Path(save_path_warped).mkdir(parents=True, exist_ok=True)

    # Get data
    x, _ = dataset[index]
    
    # Convert tensors to numpy arrays for saving
    image1 = x.numpy()[0] * 127.5 + 127.5
    image2 = x.numpy()[1] * 127.5 + 127.5

    # Resize images
    image1_resized = cv2.resize(image1, target_size, interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, target_size, interpolation=cv2.INTER_AREA)

    # Save images as PNG
    cv2.imwrite(f"{save_path_original}/image_{index}.png", image1_resized)
    cv2.imwrite(f"{save_path_warped}/image_{index}.png", image2_resized)

####################
# MAIN CODE
####################

# Define Paths
train_path = 'D:\Classical and Deep Learning Approaches for Geometric Computer Vision\Image_Stitching\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Train\Trainprocessed'  # Replace with your train dataset path
validation_path = 'D:\Classical and Deep Learning Approaches for Geometric Computer Vision\Image_Stitching\YourDirectoryID_p1\YourDirectoryID_p1\Phase2\Data\Val\Valprocessed'
# Replace with your validation dataset path
save_images_path_original = 'D:\Classical and Deep Learning Approaches for Geometric Computer Vision\Project1\\rkulkarni1\YourDirectoryID_p1\Phase2\Data\\train_original'  # Replace with the path where you want to save images
save_images_path_warped = 'D:\Classical and Deep Learning Approaches for Geometric Computer Vision\Project1\\rkulkarni1\YourDirectoryID_p1\Phase2\Data\\train_warped'
# Make Dataset
TrainSet = CocoDataset(train_path)
ValidationSet = CocoDataset(validation_path)

# Save a few sample images from the dataset
for i in range(5):  # Save first 5 samples for example
    visualize_and_save_dataset(TrainSet, i, save_images_path_original, save_images_path_warped)

# Optionally print shape of the dataset
x_sample, y_sample = TrainSet[0]
print("Shape of image pair tensor:", x_sample.shape)
print("Shape of homography matrix tensor:", y_sample.shape)