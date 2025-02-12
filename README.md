# Retail: Customer Analysis System using RealSense

This project implements a customer analysis system that uses an Intel RealSense camera for various processing steps—from calibration and 3D modeling of the store to real-time face tracking and heatmap generation of intersection data. The following steps detail the entire workflow.

## Table of Contents

- [Step 1: Chessboard Calibration Tool](#step-1-chessboard-calibration-tool)
- [Step 2: Constructing the Store's 3D Model](#step-2-constructing-the-stores-3d-model)
- [Step 3: Real-Time Face Intersection Tracking, Logging, and 3D Visualization](#step-3-real-time-face-intersection-tracking-logging-and-3d-visualization)
- [Step 4: Heatmap Generation and Auto-Resizing Image Viewer](#step-4-heatmap-generation-and-auto-resizing-image-viewer)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Step 1: Chessboard Calibration Tool

This tool captures live video from an Intel RealSense camera, detects a chessboard pattern (default size: 6 columns x 4 rows), and performs calibration by computing the chessboard's rotation and translation vectors. The calibration data is then saved for later use.

### Features

- **RealSense Integration:**  
  Streams both color and depth images from an Intel RealSense camera.
  
- **Chessboard Detection:**  
  Automatically detects the chessboard pattern using OpenCV. When the pattern is detected, the live feed pauses and highlights the detected corners.  
  *(This behavior can be modified in the source code if needed.)*
  
- **Pose Estimation:**  
  Uses OpenCV's `solvePnP` method to compute the chessboard’s rotation and translation vectors.
  
- **Data Saving:**  
  Saves the color image, depth data, detected corners, and calibration parameters for further processing.
  
- **GUI Interface:**  
  Provides a user-friendly interface built with tkinter, including options for image flipping (e.g., 180°), saving data, continuing detection, and quitting the application.

### How to Run

1. **Prepare the Camera:**  
   Ensure that the camera is fixed in a position where the customer's face is clearly visible. Verify that the camera angle is correct—once calibration is complete, the camera angle cannot be changed.

2. **Run the Application:**
   python calibration.py
3. **Interface Operation:**

Live Image Streaming: The application displays a live feed from the camera.

Chessboard Detection: When the chessboard (default 6x4 pattern) is detected, the live feed pauses and the detected corners are highlighted.

180° Image Flip: Use the "Flip 180°" checkbox to rotate the image if needed.
Save Data: Press the "Save" button to store:

Color image (chessboard_rgb.png)
Depth data (chessboard_depth.npy)
Chessboard corners (chessboard_corners.npy)

Calibration parameters (rotation & translation vectors, intrinsic matrix)

Continue Detection: Press the "Continue" button to resume live streaming and clear the current detection.

Quit: Press the "Quit" button to safely stop the RealSense pipeline and exit the application.


## Step 2: Constructing the Store's 3D Model
Using the calibrated coordinate system from Step 1, this script constructs a 3D model of the store. All dimensions are in feet (ft) and the model aligns with the real-world coordinate system defined during calibration.

### Overview
**Purpose:**
To generate a 3D visualization of the store layout based on predefined geometric data.

**Model Details:**
The store elements are defined in the CUBES_INFO list. Each element (plane or wall) includes:

Name & Color: Identification and visual appearance.
Face Points: Three points defining one face of the element.
Depth Point: Used to calculate the element’s thickness.
Inversion Flags: Options to invert the x and/or y axes.
Record Face Flag: (Optional) For additional data recording if needed.
Visualization:
Uses Matplotlib’s 3D plotting tools to render each store element as a cube. Axes are labeled and each element is annotated with its name.

### How to Run
1. **Complete Calibration (Step 1):**
Ensure calibration is complete as it defines the real-world coordinate system (with one chessboard corner as the origin).
In addition to rendering the 3D model, this step also measures the distance between the store shelves (or walls) and the origin of the real-world coordinate system (the origin point established during calibration). This measurement is critical to ensure that the virtual model accurately represents the physical layout of the store and build a cube in the .py.

2. **Run the Application:**
python Store3Dmodel.py

3. **Understanding the Model:**

Axes: The plot displays the X, Y, and Z axes with appropriate limits.
Store Elements: Each plane or wall is rendered as a cube based on CUBES_INFO.
Labels: Each element’s name is shown at the centroid of its front face.
Units: All dimensions are in feet (ft) for consistency with the calibrated coordinate system.

## Step 3: Real-Time Face Intersection Tracking, Logging, and 3D Visualization

### Overview

Initializes the RealSense pipeline and retrieves color and depth frames.
Loads calibration data (rotation vectors, translation vectors, camera intrinsics, distortion coefficients) from Step 1.
Uses MediaPipe Face Mesh to detect facial landmarks in real-time.
Computes the 3D positions of selected face landmarks and defines a local coordinate system.
Utilizes a modified SORT tracker (with Kalman filtering) to assign and maintain unique IDs for detected faces.
Computes the intersection between the face’s nose vector and predefined 3D cubes (representing store walls in feet, converted to meters for computation).
Logs intersection data asynchronously into an Excel file.
Optionally displays a real-time video window (with face landmarks and nose direction) and/or a 3D Matplotlib plot showing the computed intersection and nose vector.

### How to Run

Before running this script, ensure that:

Calibration (Step 1) has been completed and the necessary calibration files are available.
Store 3D Model (Step 2) has been constructed since the cubes used for intersection computation reference the calibrated coordinate system.

Run the script using:
python  retail.py [OPTIONS]

### Command-Line Arguments 
When running the Step 3 script, you can control its behavior with the following options:

--enable_log
Enable logging of intersection data into Excel. (Default: enabled)

--enable_3d_plot
Display the 3D plot window. (Default: disabled)

--enable_display
Show the real-time video display window with overlaid tracking and detection. (Default: disabled)

--rotate180
Disable 180-degree flipping of the camera image. (By default, the image is flipped 180°; use this flag to disable that behavior.)

Example usage with display enabled:

python retail.py --enable_display --enable_3d_plot

## Step 4: Heatmap Generation and Auto-Resizing Image Viewer
In this final step, the script:

Reads Intersection Data:
Loads the Excel log file (intersection_log_1.xlsm) containing intersection records.

Generates Heatmaps:
For each wall defined in the CUBES_INFO list, it filters the data, calculates the wall's dimensions (based on its face points), and generates a density heatmap using Seaborn’s KDE plot. If available, a background image is loaded and used.

Custom Visualization:
A custom green-to-red colormap is applied to represent density, and the figure size is dynamically adjusted based on wall dimensions.

Saves Output:
The generated heatmaps (with background images) are saved to the heatmaps_with_bg folder.

Displays Images with Auto-Resize:
A Tkinter-based viewer automatically resizes the saved heatmap images to fit the window for easy review.

## How to Run

1. **Execute the final step script with:**
python heatmapsWithImage.py

The script will:

Process the Excel log file.
Generate and save heatmaps for each wall.
Launch a Tkinter window that displays the images with auto-resizing functionality.


### Troubleshooting

Calibration Issues:
Ensure the camera is properly fixed and that all calibration files exist in the data/ folder.

Missing Files:
Verify that intersection_log_1.xlsm exists in the log/ folder and that any background images are present in the images/ folder.

RealSense Issues:
Make sure your RealSense camera is connected, and no other application is using it.

Performance:
If the frame rate is low, try disabling the 3D plot or the real-time video display using command-line options.

GUI Problems:
Ensure that Tkinter is installed and working properly on your system.

### Acknowledgements

Intel RealSense SDK: For camera integration and streaming.
MediaPipe: For robust face landmark detection.
OpenCV, Matplotlib, Seaborn: For image processing and visualization.
Pandas, NumPy: For data processing.
Tkinter & Pillow: For the auto-resizing image viewer and image manipulation.


