Algoithms
1. feature based approaches - track and map feature points
2. direct appraches - use entire image without detecting feature points
3. RBG-D approaches - depth from monocular images

Steps
1. Initialization - define global coordinate system and construct initial map within that system
2. Tracking - estimate camera pose of image t+1 by solving the perspective-n-point problem
3. Mapping - expand map by computing the 3d structure for unknown regions

Additional Steps
1. Relocalization - relocalize when tracking has failed due to dast camera motion
2. Global map optimization - obtains geometrically consistent map with error estimation

Feature Base 
1. MonoSLAM (2003) vs PTAM (2007) 
2. MonoSLAM uses EKF and PTAM uses feature matching with triangulation
3. PTAM is faster due to running tracking and mapping on different threads
4. ORB-SLAM (2015) most complete feature-base monocular vSLAM approach.
5. ORB-SLAM usesBA, vison-based closed loop edetection, and 7DoF pose-graph optimization.

