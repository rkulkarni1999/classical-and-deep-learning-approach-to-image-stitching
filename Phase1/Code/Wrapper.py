#!/usr/bin/evn python

"""
RBE/CS Fall 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numba import jit
import sys
import argparse
import os

def find_feature_matches(descriptors1, descriptors2,corners1, corners2, ratio_thresh=0.4):
    """
    Find feature matches between two sets of descriptors.

    Parameters:
    descriptors1: list of np.array
        Feature descriptors from the first image.
    descriptors2: list of np.array
        Feature descriptors from the second image.
    ratio_thresh: float
        The threshold for the ratio test.

    Returns:
    matches: list of cv2.DMatch
        The list of good matches after applying the ratio test.
    """

    matches = []
    match_pairs = []
    for idx1, desc1 in enumerate(descriptors1):
        distances = [np.sum((desc1 - desc2) ** 2) for desc2 in descriptors2]
        # print(distances)
        # adsf
        # Find the two best matches
        sorted_distances = np.argsort(distances)
        best_match_idx = sorted_distances[0]
        second_best_match_idx = sorted_distances[1]

        # Apply ratio test
        if distances[best_match_idx] < ratio_thresh * distances[second_best_match_idx]:
            matches.append(cv2.DMatch(idx1, best_match_idx, distances[best_match_idx]))
            match_pairs.append([corners1[idx1],corners2[best_match_idx]])
    return matches, match_pairs

def extract_feature_descriptor(image, corners):
    """
    Extracts feature descriptors for each corner in the image.

    Parameters:
    image: np.array
        The input image.
    corners: list of tuples
        The list of corner coordinates.

    Returns:
    descriptors: list of np.array
        The list of feature descriptors.
    """
    descriptors = []
    g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_image = np.pad(g_image, ((40,40), (40,40)), mode='constant', constant_values=0)
    for x, y in corners:
        # Extract 41x41 patch centered around the corner
        # if x < 20 or y < 20 or x > image.shape[1] - 21 or y > image.shape[0] - 21:
        #     continue  # Skip corners too close to the edge
        # g_image = np.pad(g_image, ((40,40), (40,40)), mode='constant', constant_values=0)

        xi = int(x + 20)
        yi = int(y + 20)

        patch = g_image[yi:yi+40, xi:xi+40]

        blurred_patch = cv2.GaussianBlur(patch, (7,7), cv2.BORDER_DEFAULT)
        descriptor = blurred_patch[::5,::5]
        vector = descriptor.reshape((64,1))

        standardized_vector = (vector - np.mean(vector)) / vector.std()
        descriptors.append(standardized_vector)

    return descriptors

@jit
def anms(corners, corner_strength,num_retain):
    """
    Apply Adaptive Non-Maximal Suppression (ANMS) to corner points.
    """
    # corner_strengths = [corner_strength[y, x] for x, y in corners]
    # corners = [corner for _, corner in sorted(zip(corner_strengths, corners), reverse=True)]

    # retained_corners = []
    # for corner in corners:
    #     if all(np.linalg.norm(np.array(corner) - np.array(other_corner)) > radius or
    #            corner_strength[corner[1], corner[0]] > corner_strength[other_corner[1], other_corner[0]]
    #            for other_corner in retained_corners):
    #         retained_corners.append(corner)
    #         if len(retained_corners) == num_retain:
    #             break
    # print(len(retained_corners))

    inf = float('inf')
    distances = np.full((len(corners),3), inf)
    dist =0
    for i in range(len(corners)):
        for j in range(len(corners)):
            if corner_strength[corners[i][0], corners[i][1]] > corner_strength[corners[j][0], corners[j][1]]:
                # dist = np.linalg.norm(np.array(corners[i]) - np.array(corners[j]))
                dist = (corners[j][0]- corners[i][0])**2 + (corners[j][1]- corners[i][1])**2
                # dist =12
            if dist < distances[i,0]:
                distances[i,0] = dist
                distances[i,1] = corners[i][1]
                distances[i,2] = corners[i][0]

    # print(distances[:10,0])
    sorted_indices = np.argsort(-distances[:, 0])
    sorted_distances = distances[sorted_indices]
    retained_corners = [(int(x),int(y))for x,y in zip(sorted_distances[:num_retain,1],sorted_distances[:num_retain,2])]
    return retained_corners

def harris_corner_detection(image, blockSize=2, ksize=3, k=0.04):
    """
    Apply Harris Corner Detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corner_strength = cv2.cornerHarris(gray, blockSize, ksize, k)
    dst = cv2.dilate(corner_strength, None, iterations = 2)
    lm = scipy.ndimage.maximum_filter(dst, 10)
    msk = (dst == lm)
    ls = scipy.ndimage.minimum_filter(dst, 10)
    diff = ((lm-ls) > 20000)
    msk[diff == 0] = 0
    corners = np.where(msk)
    corners = list(zip(corners[0], corners[1]))  # (x, y) format
    # for i in range(len(corners)):
    #     cv2.circle(image, (corners[i][1],corners[i][0]), 3, 255, -1)

    
    # plt.imshow(image)
    # plt.show()
    # print(corners)
    return corners, corner_strength

def convert_to_keypoints(corners):
    """
    Convert a list of (x, y) corner coordinates to a list of cv2.KeyPoint objects.
    """
    return [cv2.KeyPoint(float(x),float(y), 1) for (x, y) in corners]

def ransac_homography(points1, points2, threshold=30, max_iterations=5000, inlier_ratio=0.9):

    """
    RANSAC algorithm to find homography between two sets of points.
    
    :param points1: List of (x, y) tuples from the first image.
    :param points2: List of (x, y) tuples from the second image.
    :param threshold: Distance threshold to consider a point as an inlier.
    :param max_iterations: Maximum number of iterations for RANSAC.
    :param inlier_ratio: Ratio of inliers required to stop the algorithm.
    :return: Best homography matrix.
    """
    # Convert lists of tuples to 2D numpy arrays
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)
    best_homography = None
    best_inliers = 0

    for _ in range(max_iterations):
        # Randomly select 4 points
        indices = np.random.choice(len(points1), 4, replace=False)
        selected_points1 = points1[indices]
        selected_points2 = points2[indices]

        # Compute homography
        H, _ = cv2.findHomography(selected_points1, selected_points2, method=0)

        if H is not None:
            # Transform points from the first image to the second image
            transformed_points = cv2.perspectiveTransform(np.expand_dims(points1, 0), H)[0]
            # Compute inliers
            distances = np.sqrt(np.sum((points2 - transformed_points)**2, axis=1))
            # print(distances)
            inliers_count = np.sum(distances < threshold)
            inlier_indices = np.where(distances < threshold)[0]

            # Update best homography
            if inliers_count > best_inliers:
                best_inliers = inliers_count
                best_homography = H
                inlier_points1 = points1[inlier_indices]
                inlier_points2 = points2[inlier_indices]

                # Check if we have enough inliers
                if inliers_count > inlier_ratio * len(points1):
                    break
    
    best_homography, _ = cv2.findHomography(inlier_points1, inlier_points2, method=0)
    return best_homography

def warpTwoImages(img1,img2, H):

	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel())
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel())
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin), flags = cv2.INTER_LINEAR)
	result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
	return result

def main():

    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagesPath', default="P1TestSet/Phase1/TestSet5", help='Path to the folder which contain all the images to stitch')
    Args = Parser.parse_args()
    ImagesPath = Args.ImagesPath
    if os.path.exists(ImagesPath): 
        image_list = os.listdir(ImagesPath)
        image_list.sort()
    else:
        raise Exception ("Incorrect Path")
    
    img1_path = os.path.join(ImagesPath,image_list[0])
    img1 = cv2.imread(img1_path)
    
    for i in range(1,len(image_list)):
            
        img2_path = os.path.join(ImagesPath,image_list[i])
        img2 = cv2.imread(img2_path)
        # ANMS parameters
        num_retain = 1000
        # Harris Corner Detection
        h_corner1 ,corner_strength1 = harris_corner_detection(img1)
        h_corner2, corner_strength2 = harris_corner_detection(img2)



        # Apply ANMS
        corners1 = anms(h_corner1, corner_strength1, num_retain) #(600,450)
        corners2 = anms(h_corner2, corner_strength2, num_retain)
        # print(corners1)
        # print(sadf)
        # for i in range(len(corners1)):
        #         cv2.circle(img1, (corners1[i][0],corners1[i][1]), 3, (0, 0, 255), -1)
        # cv2.imshow("anms corner", img1) 
        
        # # waits for user to press any key 
        # # (this is necessary to avoid Python kernel form crashing) 
        # cv2.waitKey(0) 
        
        # # closing all open windows 
        # cv2.destroyAllWindows() 
        # print(dfjgioj)

        # Feature
        feature_discriptor1 = extract_feature_descriptor(img1, corners1)
        feature_discriptor2 = extract_feature_descriptor(img2, corners2)
        matches,match_pairs = find_feature_matches(feature_discriptor1, feature_discriptor2,corners1, corners2, ratio_thresh=0.4)
        match_corners1 = []
        match_corners2 = []
        for pair in match_pairs:
            match_corners1.append(pair[0])
            match_corners2.append(pair[1])

        # print(corners1)
        # print(match_corners1)
        # sdf
        keypoints1 = convert_to_keypoints(corners1)
        keypoints2 = convert_to_keypoints(corners2)
        matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)

        # Draw corners on the image
        # for x, y in corners1:
        #     cv2.circle(img1, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # Display the image
        # cv2.imshow('Feature Matching', matched_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        best_homography = ransac_homography(match_corners1, match_corners2, threshold=30, max_iterations=1000, inlier_ratio=0.9)
        warped = warpTwoImages(img2,img1, best_homography)
        img1 = warped
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        # warped = cv2.resize(warped, (960, 540)) 
        # cv2.imshow('warped', warped)
        cv2.imwrite('testset4.jpg', warped) 
        cv2.resizeWindow('warped', 1024, 768)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(i)
        

if __name__ == "__main__":
    main()