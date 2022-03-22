# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from extractor import FeatureExtractor
from triangulate import Triangle
from pose import getPose

        
'''
Run SLAM algorithm on video

Inputs:
    1. path - path to mp4 video
    
Outputs:
    None
'''
def slam(video):

    # Create pyplot 3d scatter
    plt.ion()
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
        
    # Begin SLAM
    try:
        while(True):
            
            # Read next frame
            ret, frame = video.read()

            # Get previous frame keypoints
            kps0 = extractor.kps
            des0 = extractor.des
            
            # Match features from current frame to previous frame 
            matches, kps1, des1 = extractor.extract(frame)

            if matches:               
                
                # Calculate pose transformation of camera 
                matches, R, t = getPose(kps0, kps1, matches, K)
                
                # Build list of 3d world points
                points = []
                
                for match in matches:
                    
                    # Triangulte world points
                    world, mask = triangulator.triangulate(K, R, t, kps0[match.queryIdx].pt, kps1[match.trainIdx].pt)
                    
                    # Only append world point if unmasked
                    if world[:,2] > 0 and mask:
                        points.append(world)                
                    
                # Plot matches (blue --> green) on video frame
                for match in matches:
                    p1 = kps0[match.queryIdx].pt
                    p2 = kps1[match.trainIdx].pt
                    cv2.circle(frame, tuple(map(int, p1)), radius=3, color=(0,0,255))
                    cv2.circle(frame, tuple(map(int, p2)), radius=3, color=(0,255,0))
                    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), color=(0,0,255))
                         
            # Break after last frame
            if not ret:
                vid.release()
                break
                

            # Display frame with matches 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Frame', frame)
            
            # Draw 3d world points on pyplot scatter axis
            if matches:
                
                # Only interested in plotting near objects
                # Prevents scale of axis from becoming to large to view 
                points = [point for point in points if np.sqrt(point[0][0]**2+point[0][1]**2+point[0][2]**2) < 400]
                
                # [(x,y,z)] --> [x], [y], [z]
                x_points = [point[0][0] for point in points]
                y_points = [point[0][1] for point in points]
                z_points = [point[0][2] for point in points]
                
                # Plot points
                ax.scatter(x_points, y_points, z_points, s=3, c='blue', alpha=0.5)
                
                # Plot camera location as large green dot
                ax.scatter([0], [0], [0], s=50, c='green', alpha=1.0)
                
                # Show and pause
                plt.draw()
                plt.pause(5)
                
            # Wait keys for quit and pause
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(25) & 0xFF == ord('p'):
                cv2.waitKey(-1)
            
            # Reset pyplot axis
            ax.cla()

    except KeyboardInterrupt:
        vid.release()
        

# Main       
if __name__ == "__main__":
    
    # Video path
    video = cv2.VideoCapture("videos/highway.mp4")
    
    # Feature extractor
    extractor = FeatureExtractor()
    
    # Triangulator
    triangulator = Triangle()
    
    # Camera intrinsics
    F = 700
    H = 1080
    W = 1920
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]], dtype=np.float32)
    
    # Run
    slam(video)