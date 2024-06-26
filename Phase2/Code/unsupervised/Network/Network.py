"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl


# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn(prediction, labels):
    loss_fn = nn.MSELoss()
    labels = labels.float()
    loss = loss_fn(prediction, labels)
    return loss

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net()

    def forward(self,a):
        return self.model(a)
    
    def training_step(self, image_batch, label_batch):
        
        yHat = self.model(image_batch)   # make prediction
        loss = LossFn(yHat, label_batch) # compute loss
        print(f"Training Loss : {loss}") # Print loss
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, image_batch, label_batch):
        yHat = self.model(image_batch)   # make prediction
        loss = LossFn(yHat, label_batch) # compute loss
        print(f"Validation Loss : {loss}") # Print loss
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(16*16*128, 1024)
        self.fc2 = nn.Linear(1024, 8)

        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Apply the layers using F.relu for activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # Applying pooling
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # Applying pooling
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)  # Applying pooling
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout(x)

        # Flatten the output for the linear layer
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# class Net(nn.Module):
#     def __init__(self, InputSize, OutputSize):
#         """
#         Inputs:
#         InputSize - Size of the Input
#         OutputSize - Size of the Output
#         """
#         super().__init__()
#         #############################
#         # Fill your network initialization of choice here!
#         #############################
#         ...
#         #############################
#         # You will need to change the input size and output
#         # size for your Spatial transformer network layer!
#         #############################
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#         )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
#         )

#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(
#             torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
#         )

#     #############################
#     # You will need to change the input size and output
#     # size for your Spatial transformer network layer!
#     #############################
#     def stn(self, x):
#         "Spatial transformer network forward function"
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)

#         return x

#     def forward(self, xa, xb):
#         """
#         Input:
#         xa is a MiniBatch of the image a
#         xb is a MiniBatch of the image b
#         Outputs:
#         out - output of the network
#         """
#         #############################
#         # Fill your network structure of choice here!
#         #############################
#         return out
