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

                print(t)

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
    F = 500
    H = 1080
    W = 1920
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])
    slam(video)