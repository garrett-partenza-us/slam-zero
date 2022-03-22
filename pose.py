# Imports
import cv2
import numpy as np


'''
Calculate the pose transformation.

Inputs:
    1. kps0 - key points in the train image
    2. kps1 - key points in the query image
    3. matches - matches between keypoints
    
Outputs:
    1. matches - original matches
    2. R - the rotation matrix
    3. t - the translation vector
'''
def getPose(kps0, kps1, matches, K):
    
    # 8-point algorithm
    FundamentalMatrix, mask = cv2.findFundamentalMat(
        np.int32([kps0[match.queryIdx].pt for match in matches]), 
        np.int32([kps1[match.trainIdx].pt for match in matches]),
        cv2.FM_RANSAC
    )

    # Apply FundamentalMatrix mask
    matches = [match[0] for match in zip(matches, mask) if match[1] != 0]

    # Callibrate using intrinsic paramters
    EssentialMatrix = K.T.dot(FundamentalMatrix).dot(K)

    # Recover pose transformation
    good, R, t, mask = cv2.recoverPose(
        EssentialMatrix, 
        np.int32([kps0[match.queryIdx].pt for match in matches]), 
        np.int32([kps1[match.trainIdx].pt for match in matches])
    )

    # Apply EssentialMatrix mask
    matches = [match[0] for match in zip(matches, mask) if match[1] != 0]
    
    return matches, R, t