# My AutoPano
The purpose of this project is to stitch two or more images in order to create one seamless panorama image by finding the Homography between the two images. 
The project is divided into two phases,
1. Phase 1: Classical approach of local feature matching
2. Phase 2: Deep Learning approach(Homography Net - supervised and unsupervised) to estimate the homography.

### Team Members

- Rutwik Kulkarni (rkulkarni1@wpi.edu)
- Ankit Mittal (amittal@wpi.edu)

## **Phase 1 - Using Classical CV **
Implemented traditional CV pipeline combines algorithms of corner detection, ANMS, feature extraction,
feature matching, RANSAC, homography estimation and blending.

### Results

#### Corner Detection and Non Maximal Suppression
<img src="./Phase1/Code/results/harris_corner_detector.png"  align="center" alt="Undistorted" width="500"/>

#### Feature Matching
<img src="./Phase1/Code/results/feature_matching.png"  align="center" alt="Undistorted" width="500"/>

#### Outlier Rejection using RANSAC
<img src="./Phase1/Code/results/outlier_rejection_ransac.png"  align="center" alt="Undistorted" width="500"/>

#### Warping, Blending and Stitching
<img src="./Phase1/Code/Results/final1.png"  align="center" alt="Undistorted" width="500"/>

<img src="./Phase1/Code/Results/final2.png"  align="center" alt="Undistorted" width="500"/>

<img src="./Phase1/Code/Results/final3.png"  align="center" alt="Undistorted" width="500"/>

### Usage Guidelines

1. Open directory Phase1/Code and run the following command with the Data location as command line argument: -

    ```
    python3 Wrapper.py 
    ```

2. `Results` folder contains stitched images of all Train Sets and Test Sets.

## **Phase 2 - using Deep Learning**
In Deep learning, used Homography Net (both supervised and unsupervised) to estimate the homography.

### DataSet Generation
To generate dataset, run the following command in Phase2/Code/supervised: -
    ```
    python3 make_dataset.py
    ```

#### Original Patch

<img src="./Phase2/Code/supervised/Results/original_patch.jpg"  align="center" alt="Undistorted" width="400"/>

#### Patch after applying perturbations 

<img src="./Phase2/Code/supervised/Results/warped_patch.jpg"  align="center" alt="Undistorted" width="400"/>

### Supervised Homography

<img src="./Phase2\Code\supervised\Results\network_architecture_supervised.png"  align="center" alt="Undistorted" width="400"/>

#### Result

##### Training Loss
<img src="./Phase2/Code/supervised/Results/train_loss_sup.jpg"  align="center" alt="Undistorted" width="300"/>

##### Validation Loss
<img src="./Phase2/Code/supervised/Results/val_loss_sup.jpg"  align="center" alt="Undistorted" width="300"/>

##### Stitched Image. 

<img src="./Phase2/Code/supervised/Results/Testset2pano.png"  align="center" alt="Undistorted" width="300"/>

<img src="./Phase2/Code/supervised/Results/Testset3pano.png"  align="center" alt="Undistorted" width="300"/>

1. To train the network, run: -
    ```
    python3 Train.py
    ```

2. To test the model on test set, run: -
    ```
    python3 Test.py
    ```

### Unsupervised Homography

<img src="./Phase2/Code/unsupervised/Unsupervised.png"  align="center" alt="Undistorted" width="550"/>

1. To train the network, run: -
    ```
    python3 Train.py
    ```

2. To test the model on trainset , run: -
    ```
    python3 Test.py
    ```

#### Result

##### Training Loss

<img src="./Phase2/Code/unsupervised/Results/val_loss_unsup.jpg"  align="center" alt="Undistorted" width="300"/>

##### Validation Loss

<img src="./Phase2/Code/unsupervised/Results/val_loss_unsup.jpg"  align="center" alt="Undistorted" width="300"/>


##### Output on the Patch (Image, Ground Truth, Prediction)

<img src="./Phase2/Code/unsupervised/Results/t2.jpg"  align="center" alt="Undistorted" width="300"/>