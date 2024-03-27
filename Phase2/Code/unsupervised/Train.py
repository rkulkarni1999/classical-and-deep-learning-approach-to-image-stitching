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
# from Network.Network import HomographyModel
from Network.UN_Network import HomographyModel
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
# from Network.Network import LossFn 
from random import choice
import torchvision.transforms as transforms
import multiprocessing


#######
# GPU
#######
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ######################
# # Defining Transforms
# ######################
# transform = transforms.Compose([
#     transforms.ToPILImage(),            # Convert numpy array to PILImage
#     transforms.Resize((128, 128)),      # Resize to a fixed size
#     transforms.ToTensor(),              # Convert PILImage to PyTorch Tensor
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
# ])

# ##################################################
# # Function for loading .npy files to dictonaries
# ##################################################
# def load_labels(label_dir):
#     labels = {}
#     for file in os.listdir(label_dir):
#         if file.endswith(".npy"):
#             file_path = os.path.join(label_dir, file)
#             image_id = 'image_' + file.replace('label_', '').replace('.npy', '')
#             labels[image_id] = np.load(file_path)
#     return labels

#############################
# Function for DirNamesTrain 
#############################
def create_dir_names_train(folder_path):
    all_files = os.listdir(folder_path)
    dir_names_train = [file for file in all_files if file.endswith('.jpg')]
    return dir_names_train

# def GenerateBatch(BasePath, DirNamesTrain, DirNamesVal, TrainCoordinates, ValCoordinates, ImageSize, MiniBatchSize, dataset):

#     return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)


def PrettyPrint(Args, LatestFile,NumTrainSamples):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(Args.epochs))
    print("Factor of reduction in training data is " + str(Args.DivTrain))
    print("Mini Batch Size " + str(Args.batch_size))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

def TrainOperation(Args, LatestFile,NumTrainSamples):
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
    model = HomographyModel(model_hparams=Args)
    model = model.to(device)

    # Optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum = 0.9)
    # Optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    Optimizer = model.configure_optimizers()
    
    # Initialize TensorBoard Writer
    Writer = SummaryWriter(Args.LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(Args.CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    epoch_losses = []

    for Epochs in tqdm(range(StartEpoch, Args.epochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / Args.batch_size / Args.DivTrain)
        total_loss = 0
        model.train()
        Train_Dataloder = model.train_dataloader()
        val_Dataloader = model.val_dataloader()

        PerEpochCounter = 0
        for Batch in Train_Dataloder:
            # I1Batch, CoordinatesBatch = GenerateBatch(
            #     BasePath, DirNamesTrain,DirNamesVal, TrainCoordinates,ValCoordinates,ImageSize, MiniBatchSize,"Train"
            # )
            
            # CoordinatesBatch = CoordinatesBatch.view(CoordinatesBatch.size(0), -1)
            
            # # Predict output with forward pass
            # PredicatedCoordinatesBatch = model.(I1Batch)
            # LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            # total_loss += LossThisBatch.item()
            PerEpochCounter += 1
            LossThisBatch = model.training_step(Batch)
            total_loss += LossThisBatch
            
            # print(f"{PerEpochCounter} Batch Loss : {LossThisBatch}")
            
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            SaveCheckPoint = 100
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                
                # Write training losses to tensorboard
                Writer.add_scalar(
                    "LossEveryIter",LossThisBatch,Epochs * NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
                # Save the Model learnt in this epoch
                SaveName = (
                    Args.CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

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
        
        # Calculate and print average training loss for the epoch
        avg_training_loss = total_loss / NumIterationsPerEpoch
        epoch_losses.append(avg_training_loss)
        print(f"Average Training Loss for Epoch {Epochs}: {avg_training_loss}")

        # Log the average training loss for this epoch
        Writer.add_scalar("Loss/Train", avg_training_loss, Epochs)
        
        model.eval()
        val_loss =0
        with torch.no_grad():
            # val_batch, val_labels = GenerateBatch(BasePath, DirNamesVal,ValCoordinates,ImageSize, MiniBatchSize,"Val")
            # val_batch, val_labels = GenerateBatch(BasePath,DirNamesTrain,DirNamesVal, TrainCoordinates,ValCoordinates, ImageSize,MiniBatchSize,"Val")
            # val_labels = val_labels.view(val_labels.size(0), -1)
            for val_batch in val_Dataloader:
                val_lossthisbatch = model.validation_step(val_batch)
                val_loss += val_lossthisbatch["val_loss"]
            avg_validation_loss = val_loss / len(val_Dataloader)
            Writer.add_scalar("Loss/Val", avg_validation_loss, Epochs)
            print(f"validationloss : {avg_validation_loss}")
        # Write validation losses to tensorboard
        # Writer.add_scalar(
        #     "LossEveryEpoch",
        #     result["val_loss"],
        #     Epochs,
        # )
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()
        
        # Save model every epoch
        if(Epochs%20 ==0):
            SaveName = Args.CheckPointPath + str(Epochs) + "model.ckpt"
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
        "--train_path",
        default="D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\\rkulkarni1\Phase2\Data\Train\Train",
        help="Base path of images",
    )
    Parser.add_argument(
        "--valid_path",
        default="D:\Study\RBE 549\Project1\computer_vision_hw\computer_vision_hw\\rkulkarni1\Phase2\Data\Val\Val",
        help="Base path of images",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpointss/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=1,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    Parser.add_argument(
        "--rho", 
        type=int, 
        default=45, help="amount to perturb corners")

    Parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate")

    Args = Parser.parse_args()

    # Find Latest Checkpoint File
    if Args.LoadCheckPoint == 1:
        LatestFile = FindLatestModel(Args.CheckPointPath)
    else:
        LatestFile = None
    
    DirNamesTrain = create_dir_names_train(Args.train_path)
    NumTrainSamples = len(DirNamesTrain)
    PrettyPrint(Args,LatestFile,NumTrainSamples)
    TrainOperation(Args, LatestFile,NumTrainSamples)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
