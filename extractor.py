# Imports
import cv2
import numpy as np

# Used to extract matches between current and previous frames
class FeatureExtractor():
    
    def __init__(self):
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.frame = np.array([])
        self.kps = np.array([])
        self.des = np.array([])
        
    def filterMatches(self, matches):
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good
    
    def extract(self, frame):
        
        features = cv2.goodFeaturesToTrack(
                       np.mean(frame, axis=2).astype(np.uint8),
                       maxCorners=3000,
                       qualityLevel=0.01,
                       minDistance=3
                   )
        
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
        kps, des  = self.orb.compute(frame, kps)
        matches = None
        
        if self.frame.any():
            matches = self.bf.knnMatch(self.des, des, k=2)
            matches = self.filterMatches(matches)
                                
        self.frame = frame
        self.kps = kps
        self.des = des

        return matches, kps, des