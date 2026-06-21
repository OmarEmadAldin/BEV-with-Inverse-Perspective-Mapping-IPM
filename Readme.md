# bird's-eye view (BEV) with Inverse Perspective Mapping (IPM)
 
Inverse Perspective Mapping (IPM) transforms standard, perspective-distorted 2D images into a perpendicular, 
top-down Bird's Eye View (BEV). 
It is widely used in autonomous vehicles and robotics to measure metric distances on the ground


## End-to-End Pipeline

    Video 
       ↓
    YOLOv8 Detection
       ↓
    Ground Contact Extraction (bottom-center of box)
       ↓
    Homography (IPM) (better to use real data with real camera calib files to make it instead of built in func)
       ↓
    Point Projection to BEV
       ↓
    Visualization

## Mathematical Concept
To transition from a 2D image plane to a 2D ground plane, a series of 
geometric assumptions and matrix multiplications are performed. Because the camera and world both operate in 3D, IPM requires "lifting" the image coordinates to 3D, 
rotating/translating them, and then projecting them onto a flat surface.
- Intrinsic Parameters: Camera properties like focal length and lens distortion.

- Extrinsic Parameters: The physical position and angle of the camera relative to the ground.

- Flat Earth Assumption: IPM assumes the road is perfectly flat. Points from the camera's image plane are projected onto this 2D ground plane (typically the z = 0 axis)



## Codes Functions

### config.py
-   `Contains all the constants and the parameters `
### IPM.py

-   `compute_homography(src_pts, dst_pts)`
-   `warp_to_bev(frame, H, size)`
-   `project_point((u, v), H)`

### detector.py

-   `load_yolo()`
-   `detect(frame) -> List[Tuple]`

### visualization.py

-   `draw_perspective(frame, detections)`
-   `draw_bev(bev_frame, detections, H)`
-   `compose(persp, bev)`

### main.py

-   `process_video(video_path, H, out_path, max_frames)`

## Data Used for test

        wget - https://www.kaggle.com/datasets/eeemmm1234/kitti-track-video?select=0000.mp4

        wget - https://www.kaggle.com/datasets/eeemmm1234/kitti-track-video?select=0004.mp4


## Results

<p align="center">

  <img src="output/vid_1.gif" width="100%" />
</p>

<p align="center">

  <img src="output/vid_2.gif" width="100%" />
</p>


