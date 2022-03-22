# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from extractor import FeatureExtractor


# Get the pose transformation from keypoint matches
def getPose(kps0, kps1, matches):
    # 8-point algorithm
    FundamentalMatrix, mask = cv2.findFundamentalMat(
        np.int32([kps0[match.queryIdx].pt for match in matches]), 
        np.int32([kps1[match.trainIdx].pt for match in matches]),
        cv2.FM_RANSAC
    )

    # Apply FundamentalMatrix mask
    matches = [match[0] for match in zip(matches, mask) if match[1] != 0]

    # Callibrate
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

class Triangle():
    
    def __init__(self):
        # First frame origin
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        self.t = np.array([[0],[0],[0]], dtype=np.float32)
        
    def triangulate(self, K, R, t, x, y):
        # Triangulate
        Rt1 = np.hstack([self.R.T, -self.R.T.dot(self.t)])
        projection1 = np.dot(K, Rt1)
        Rt2 = np.hstack([R.T, -R.T.dot(t)])
        projection2 = np.dot(K, Rt2)
        imagePoint1, imagePoint2 = x, y
        point3D = cv2.triangulatePoints(projection1, projection2, imagePoint1, imagePoint2).T
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
        

        
# Run slam on a video path
def slam(video):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    
#     location = np.array([0,0,0])
    
    try:
        while(True):
            
            # Read next frame
            ret, frame = video.read()

            # Get previous frame keypoints
            kps0 = extractor.kps
            des0 = extractor.des
            
            # Feature map current frame to last frame 
            matches, kps1, des1 = extractor.extract(frame)

            if matches:               
                
                # Pose transformation
                matches, R, t = getPose(kps0, kps1, matches)
                # Triangulation
                points = []
                for match in matches:
                    world, mask = triangulator.triangulate(K, R, t, kps0[match.queryIdx].pt, kps1[match.trainIdx].pt)
                    if world[:,2] > 0 and mask:
                        points.append(world)                
                    
                # Plot on frame
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
                

            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow('Frame', frame)
            
            if matches:
                # specify the location of (left,bottom),width,height
                import matplotlib.patches as mpatches
                rect=mpatches.Rectangle((0,0),14,7, 
                                        fill = False,
                                        color = "purple",
                                        linewidth = 2)
                points = [point for point in points if np.sqrt(point[0][0]**2+point[0][1]**2+point[0][2]**2) < 400]
                x_points = [point[0][0] for point in points]
                y_points = [point[0][1] for point in points]
                z_points = [point[0][2] for point in points]
                ax.scatter(x_points, y_points, z_points, s=3, c='blue', alpha=0.5)
                ax.scatter([0], [0], [0], s=50, c='green', alpha=1.0)
                plt.draw()
                plt.pause(5)
                
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(25) & 0xFF == ord('p'):
                cv2.waitKey(-1)
            
            ax.cla()

    except KeyboardInterrupt:
        vid.release()
        
# Main
if __name__ == "__main__":
    
    # Video path
    video = cv2.VideoCapture("highway.mp4")
    # Feature extractor
    extractor = FeatureExtractor()
    # Triangulator
    triangulator = Triangle()
    # Camera intrinsics
    F = 700
    H = 1080
    W = 1920
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]], dtype=np.float32)
    slam(video)
    
                    # Convert to normalized homogenous 
#                 pts1 = np.int32([kps0[match.queryIdx].pt for match in matches]) 
#                 pts2 = np.int32([kps1[match.trainIdx].pt for match in matches])
#                 points1 = [np.hstack([point, [1]]) for point in pts1]
#                 points2 = [np.hstack([point, [1]]) for point in pts2]
#                 points1_norm = np.array([np.dot(np.linalg.inv(K), point) for point in points1])
#                 points2_norm = np.array([np.dot(np.linalg.inv(K), point) for point in points2])
#                 # Compute pose
#                 p1n, T1 = scale_and_transform_points(points1_norm)
#                 p2n, T2 = scale_and_transform_points(points2_norm)
#                 # Compute F
#                 F = compute_img_to_img_matrix(p1n, p2n, False)
#                 F = np.dot(T1.T, np.dot(F, T2))
#                 E = F / F[2, 2]
#                 P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
#                 P2s = compute_P_from_essential(E)
#                 ind = -1
#                 for i, P2 in enumerate(P2s):
#                     d1 = reconstruct_one_point(points1_norm[:, 0], points2_norm[:, 0], P1, P2)
#                     P2_homogenous = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))
#                     d2 = np.dot(P2_homogenous[:3, :4], d1)
#                     if d1[2] > 0 and d2[2] > 0:
#                         ind = i
#                 P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
#                 # Triangulate
#                 tripoints3d = triangulation(points1_norm, points2_norm, P1, P2)
#                 x_points = [pt for pt in tripoints3d[0]]
#                 y_points = [-pt for pt in tripoints3d[1]]
#                 z_points = [-pt for pt in tripoints3d[2]]