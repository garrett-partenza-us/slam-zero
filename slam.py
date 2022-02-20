# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from extractor import FeatureExtractor


# Get the pose transformation from keypoint matches
def getPose(kps0, kps1, matches):
    FundamentalMatrix, mask = cv2.findFundamentalMat(
        np.int32([kps0[match.queryIdx].pt for match in matches]), 
        np.int32([kps1[match.trainIdx].pt for match in matches]),
        cv2.FM_RANSAC
    )

    matches = [match[0] for match in zip(matches, mask) if match[1] != 0]

    EssentialMatrix = K.T.dot(FundamentalMatrix).dot(K)

    good, R, t, mask = cv2.recoverPose(
        EssentialMatrix, 
        np.int32([kps0[match.queryIdx].pt for match in matches]), 
        np.int32([kps1[match.trainIdx].pt for match in matches])
    )

    matches = [match[0] for match in zip(matches, mask) if match[1] != 0]
    
    return matches, R, t

class Triangle():
    
    def __init__(self):
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        self.t = np.array([[0],[0],[0]], dtype=np.float32)
        
    def triangulate(self, K, R, t, x, y):
        Rt1 = np.hstack([self.R.T, -self.R.T.dot(self.t)])
        projection1 = np.dot(K, Rt1)
        Rt2 = np.hstack([R.T, -R.T.dot(t)])
        projection2 = np.dot(K, Rt2)
        imagePoint1, imagePoint2 = x, y# cv2.undistortPoints(x, K, None, projection1), cv2.undistortPoints(y, K, None, projection2)
        point3D = cv2.triangulatePoints(projection1, projection2, imagePoint1, imagePoint2).T
        point3D = point3D[:, :3] / point3D[:, 3:4]
        print(point3D)
        # Reproject back into the two cameras
        rvec1, _ = cv2.Rodrigues(self.R.T) # Change
        rvec2, _ = cv2.Rodrigues(R.T) # Change
        p1, _ = cv2.projectPoints(point3D, rvec1, -self.t, K, distCoeffs=None) # Change
        p2, _ = cv2.projectPoints(point3D, rvec2, -t, K, distCoeffs=None) # Change

        reprojection_error1 = np.linalg.norm(x - p1[0, :])
        reprojection_error2 = np.linalg.norm(y - p2[0, :])
        
#         print(x, " -> ", p1, " ... Error : ", reprojection_error1)
#         print(y, " -> ", p2, " ... Error : ", reprojection_error2)
        return point3D
        
# Run slam on a video path
def slam(video):
    try:
        while(True):

            ret, frame = video.read()

            kps0 = extractor.kps
            des0 = extractor.des
            matches, kps1, des1 = extractor.extract(frame)
            
            if matches:
                
                matches, R, t = getPose(kps0, kps1, matches)
                points = []
                for match in matches:
                    world = triangulator.triangulate(K, R, t, kps0[match.queryIdx].pt, kps1[match.trainIdx].pt)
                    points.append(world)
                triangulator.R = R
                triangulator.t = t
                    
                for match in matches:
                    p1 = kps0[match.queryIdx].pt
                    p2 = kps1[match.trainIdx].pt
                    cv2.circle(frame, tuple(map(int, p1)), radius=3, color=(0,0,255))
                    cv2.circle(frame, tuple(map(int, p2)), radius=3, color=(0,255,0))
                    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), color=(0,0,255))
                                    
            if not ret:
                vid.release()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(25) & 0xFF == ord('p'):
                cv2.waitKey(-1)

    except KeyboardInterrupt:
        vid.release()
        
# Main
if __name__ == "__main__":
    
    video = cv2.VideoCapture("highway.mp4")
    extractor = FeatureExtractor()
    triangulator = Triangle()
    F = 700
    H = 1080
    W = 1920
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]], dtype=np.float32)
    slam(video)