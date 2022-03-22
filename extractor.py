# Imports
import cv2
import numpy as np


'''
Class to extract feature points. Necessary to
maintain the features of last frame for matching
to current frame.
'''
class FeatureExtractor():
    
    '''
    Initialization function to maintain the previous
    feature points and descriptors.
    
    Inputs:
        1. self - self
        
    Outputs:
        None
    '''
    def __init__(self):
        
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.frame = np.array([])
        self.kps = np.array([])
        self.des = np.array([])
        
    '''
    Filters unconfident matches with ratio test
    
    Input2:
        1. matches - list of point correspondances
    
    Outputs:
        1. good - matches with bad ones dropped
    '''
    def filterMatches(self, matches):
        
        # Ratio test @ 0.75
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good
    
    '''
    Extract features of the current frame and match
    to the previous frame.
    
    Inputs:
        1. frame - RBG image frame
        
    Outputs:
        1. matches - matches between points in previous and current imgage
        2. kps - keypoints in current image
        3. descriptors of keypoints in current image
    '''
    def extract(self, frame):
        
        # Calculate ORB features of current frame
        features = cv2.goodFeaturesToTrack(
                       np.mean(frame, axis=2).astype(np.uint8),
                       maxCorners=3000,
                       qualityLevel=0.01,
                       minDistance=3
                   )
        
        # Get keypoints from features 
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        
        # Get descriptors
        kps, des  = self.orb.compute(frame, kps)
        
        # Initialize matches
        matches = None
        
        # Calculate matches using brute force
        if self.frame.any():
            matches = self.bf.knnMatch(self.des, des, k=2)
            matches = self.filterMatches(matches)
                      
                
        # Remember class frame, keypoints and descriptors for next frame
        self.frame = frame
        self.kps = kps
        self.des = des

        return matches, kps, des