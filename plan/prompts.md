Now I would like you to create a new script, ned2_pose_viz_video.py, which will load an mp4 video and then run the model on the video and save it in the same folder as script is. Please use the supervision parse and annotator. Please maek the first target [best_video_normal_segment_006.mp4](ned2_pose/best_video_normal_segment_006.mp4) 

Currently the MCMOT object has three displays, one for each camera and then an overhead map plot. I want to add a fourth display, that is a textured mesh of the scene. To create the rendering, you will need to use opencv stereoRectify method to get a dense depth map, the reprojectImageTo3D method to get 3D points. Then turn the point cloud into a mesh with openHere is a pipline that shows what should happen.

Calibrate each camera
Use OpenCV to get intrinsics and distortion, then undistort both images.

Estimate metric camera poses with landmarks
Use known 3D landmarks and their 2D image locations with solvePnPRansac so both cameras are placed in the same real-world coordinate system.

Compute the relative camera transform
Derive the rotation and translation between the two cameras from their world poses.

Rectify the images
Use stereoRectify so corresponding points line up for dense matching.

Create dense depth
Use StereoSGBM to compute disparity, then reprojectImageTo3D to convert disparity into a 3D point cloud.

Create a surface mesh
Build an Open3D point cloud, estimate normals, then run create_from_point_cloud_poisson to reconstruct a surface mesh.

Clean the mesh
Remove low-density Poisson regions, recompute normals, and optionally smooth or decimate.

Texture the mesh
Project the original camera image onto the mesh so faces are colored from the source imagery.

Export for the GUI
Save as glTF/GLB for display in a Vue app with Three.js.



Ok, the current set_extrinsics relies on the aruco markers. I want to keep that intact and then but also add another way to calibrate the cameras, just by clicking on known landmarks in each frame. When the MCMOT class is initalized, I want an option to be calibration, the options are aruco or landmarks. That argument is then passed to the camera objects, and used in the set_extrinsics. Put the existing aruco calibration in its own method camera.cal_aruco(), and then create a new method camera.cal_landmarks(). The calibration process should open a screen that prompts the user to click on a landmark, and then enter a known coordinate. The user user shoudl be able to click space bar or enter to enter a point, and then c to complete the calibration.

Review all the files in the the source folder and create a mermaid class diagram that shows the relationships between the classes. The class diagram should include the class names, their attributes, and their methods. Additionally, it should indicate the inheritance relationships between the classes, as well as any associations or dependencies. The diagram should be clear and easy to understand, providing a visual representation of the structure of the codebase.