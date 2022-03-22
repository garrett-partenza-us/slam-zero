# Imports
import cv2
import numpy as np


'''
Triangulator class for triangulating world points
'''
class Triangle():
    
    '''
    Initialization function to maintain the previous
    R and t matricies
    
    Inputs:
        1. self - self
        
    Outputs:
        None
    '''
    def __init__(self):
        
        # First frame origin
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        self.t = np.array([[0],[0],[0]], dtype=np.float32)
    
    '''
    Triangulate real world points of features in an image.
    Masks points with large reprojection errors.
    
    Inputs:
        1. self - self
        2. K - intrinisc camera parameters
        3. R - the current rotation matriix
        4. t - the current translation matrix
        5. x - image coordinates in previous image 
        6. y - image coordinates in current image 
        
    Outputs:
        1. point3d - 3d diminension world point with 
           origin at (0,0,0)
        2. mask - binary mask on reprojection error
    '''
    def triangulate(self, K, R, t, x, y):
        
        # Projecftion matirx one
        Rt1 = np.hstack([self.R.T, -self.R.T.dot(self.t)])
        projection1 = np.dot(K, Rt1)
        
        # Projecftion matrix two
        Rt2 = np.hstack([R.T, -R.T.dot(t)])
        projection2 = np.dot(K, Rt2)
        
        imagePoint1, imagePoint2 = x, y
        
        # Calculate 4d world point
        point3D = cv2.triangulatePoints(projection1, projection2, imagePoint1, imagePoint2).T
        
        # Map 4d --> 3d
        point3D = point3D[:, :3] / point3D[:, 3:4]
        
        # Reproject back into the two cameras
        rvec1, _ = cv2.Rodrigues(self.R.T) 
        rvec2, _ = cv2.Rodrigues(R.T) 
        p1, _ = cv2.projectPoints(point3D, rvec1, -self.t, K, distCoeffs=None) 
        p2, _ = cv2.projectPoints(point3D, rvec2, -t, K, distCoeffs=None) 
        
        # Get the reprojection errror
        reprojection_error1 = np.linalg.norm(x - p1[0, :])
        reprojection_error2 = np.linalg.norm(y - p2[0, :])
        
        # Mask reprojection-erroneous points
        if reprojection_error1 <= 2 and reprojection_error2 <= 2:
            return point3D, 1
        else:
            return point3D, 0