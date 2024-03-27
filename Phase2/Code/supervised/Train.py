#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
# from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Network.Network import LossFn 
from random import choice
import torchvision.transforms as transforms
# from create_validation_loader import validation_dataloader

#######
# GPU
#######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

######################
# Defining Transforms
######################
transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert numpy array to PILImage
    # transforms.Resize((128, 128)),      # Resize to a fixed size
    transforms.ToTensor(),              # Convert PILImage to PyTorch Tensor
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
])

##################################################
# Function for loading .npy files to dictonaries
##################################################
def load_labels(label_dir):
    labels = {}
    for file in os.listdir(label_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(label_dir, file)
            image_id = 'image_' + file.replace('label_', '').replace('.npy', '')
            labels[image_id] = np.load(file_path)
    return labels

#############################
# Function for DirNamesTrain 
#############################
def create_dir_names_train(folder_path):
    all_files = os.listdir(folder_path)
    dir_names_train = [file for file in all_files if file.endswith('.png')]
    return dir_names_train

def GenerateBatch(BasePath, DirNamesTrain, DirNamesVal, TrainCoordinates, ValCoordinates, ImageSize, MiniBatchSize, dataset):
    """
    Generate a batch of data for training or validation.

    Parameters:
    - BasePath: Path to the base folder containing the dataset.
    - DirNamesTrain: List of training image filenames.
    - DirNamesVal: List of validation image filenames.
    - TrainCoordinates: Dictionary of training image coordinates.
    - ValCoordinates: Dictionary of validation image coordinates.
    - ImageSize: Size to which the images are to be resized.
    - MiniBatchSize: The size of the batch to generate.
    - dataset: A string indicating whether the batch is for 'Train' or 'Val'.
    - selected_indices: Indices of specific images to use (for validation set).

    Returns:
    - Batch of images and corresponding coordinates.
    """
    I1Batch = []
    CoordinatesBatch = []

    if dataset == "Train":
        image_names = DirNamesTrain
        coordinates = TrainCoordinates
    elif dataset == "Val":
        image_names = DirNamesVal
        coordinates = ValCoordinates
    else:
        raise ValueError("Invalid dataset specified. Choose 'Train' or 'Val'.")
        
    indices = random.sample(range(len(image_names)), MiniBatchSize) # random indices for choosing images in MiniBatch

    for idx in indices:
        
        RandImageName = image_names[idx]
        image_id = RandImageName.split('.')[0]

        original_image_path = os.path.join(BasePath, f"{dataset}/{dataset}_original_multiple", RandImageName)
        warped_image_path = os.path.join(BasePath, f"{dataset}/{dataset}_warped_multiple", RandImageName)

        I1_original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        I1_warped   = cv2.imread(warped_image_path, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded properly
        if I1_original is None or I1_warped is None:
            print(f"Warning: Unable to load image at {original_image_path} or {warped_image_path}")
            continue 

        # Apply transformations
        I1_original = transform(I1_original)
        I1_warped   = transform(I1_warped)

        # concatenate depth wise. 
        I1 = np.concatenate([I1_original, I1_warped], axis=0)
        I1 = np.float32(I1)
        
        # extract label from image_id
        Coordinate = coordinates[image_id]

        # Append All Images and Coordinates
        I1Batch.append(torch.from_numpy(I1))
        CoordinatesBatch.append(torch.tensor(Coordinate))

    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

def TrainOperation(
    DirNamesTrain,
    DirNamesVal,                
    TrainCoordinates,
    ValCoordinates,               
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()
    model = model.to(device)

    # Optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum = 0.9)
    Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Initialize TensorBoard Writer
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    epoch_losses = []

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        
        total_loss = 0 
        
        ############
        # TRAINING
        ############
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatch(
                BasePath, DirNamesTrain,DirNamesVal, TrainCoordinates,ValCoordinates,ImageSize, MiniBatchSize,"Train"
            )

            CoordinatesBatch = CoordinatesBatch.view(CoordinatesBatch.size(0), -1)
            
            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            total_loss += LossThisBatch.item()
            
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                
                # Write training losses to tensorboard
                Writer.add_scalar(
                    "LossEveryIter",LossThisBatch,Epochs * NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )
    
        # Calculate and print average training loss for the epoch
        avg_training_loss = total_loss / NumIterationsPerEpoch
        epoch_losses.append(avg_training_loss)
        print(f"Average Training Loss for Epoch {Epochs}: {avg_training_loss}")

        # Log the average training loss for this epoch
        Writer.add_scalar("Loss/Train", avg_training_loss, Epochs)
        
        ###############
        # Validation 
        ###############
        model.eval()
        val_batch, val_labels = GenerateBatch(BasePath,DirNamesTrain,DirNamesVal, TrainCoordinates,ValCoordinates, ImageSize,256,"Val")
        val_labels = val_labels.view(val_labels.size(0), -1)
        pred = model(val_batch)
        val_loss = LossFn(pred, val_labels)
        print(f"Validation Loss : {val_loss.item()}") # Print loss
        result = {"val_loss": val_loss}
        Writer.add_scalar("Loss/Val", result["val_loss"], Epochs)
        
        Writer.flush()
        # val_loss = 0
        # model.eval()
        # for val_batch, val_labels in validation_dataloader:
        #     val_batch = val_batch.to(device)
        #     val_labels = val_labels.to(device)
        #     pred = model(val_batch)
        #     val_loss += LossFn(pred, val_labels).item()
        # avg_val_loss = val_loss / (len(validation_dataloader) * 100) # divided by num of data points in val loader
        # print(f"Validation Loss: {avg_val_loss}")
        # result = {"val_loss": avg_val_loss}
        # Writer.add_scalar("Loss/Val", result["val_loss"], Epochs)
        # Writer.flush()

        # Save checkpoint every 20 iterations
        # # Save model every epoch
        if (not(Epochs < 20)) and (Epochs % 20 == 0):

            SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
            torch.save(
                {
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": LossThisBatch,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")
    
    Writer.close()
    
def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/pear_group/rutwik/computer_vision_hw/rkulkarni1/Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints_trans/",
        help="Path to save Checkpoints_trans, Default: ../Checkpoints_trans/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=200,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        SaveCheckPoint,
        ImageSize,
    ) = SetupAll(BasePath, CheckPointPath)

    checkpoint_file_name = '100model.ckpt'
    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath=CheckPointPath, file=checkpoint_file_name)
    else:
        LatestFile = None

    # Load labels from Train and Validation Sets from .npy files to Dictionary 
    TrainCoordinates = load_labels(f'{BasePath}/Train/TrainLabels')
    ValCoordinates   = load_labels(f'{BasePath}/Val/ValLabels')
        
    # Path to Train and Val Images. 
    path_train_images = f"{BasePath}/Train/Train_original_multiple"  
    path_val_images   = f"{BasePath}/Val/Val_original_multiple"
    
    # Create DirNamesTrain 
    DirNamesTrain = create_dir_names_train(path_train_images) # List of all image names in Train set. 
    DirNamesVal   = create_dir_names_train(path_val_images)
    
    print(f"DirnamesTrain : {len(DirNamesTrain)}")
    print(f"DirnamesVal   : {len(DirNamesVal)}")
    
    NumTrainSamples = len(DirNamesTrain)
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)


    TrainOperation(
        DirNamesTrain,
        DirNamesVal,
        TrainCoordinates,
        ValCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )

if __name__ == "__main__":
    main()
