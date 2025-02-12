import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Requires 'pillow' package


# -------------------------------------
# Chessboard-related functions
# -------------------------------------
def generate_object_points(checkerboard_size, square_size):
    """
    Generate 3D coordinates for the chessboard in the world coordinate system (Z=0).
    checkerboard_size: (cols, rows)
    square_size: The length of each square on the chessboard (in meters).
    """
    objp = np.zeros((checkerboard_size[1] * checkerboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def estimate_pose(objp_world, imgp, rgb_intrinsic_matrix, dist_coeffs=None):
    """
    Estimate the chessboard pose (rvec, tvec) using solvePnP.
    - objp_world: (N, 3) 3D points in world coordinates
    - imgp: (N, 2) 2D points (pixel coordinates)
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    if len(objp_world) < 4 or len(imgp) < 4:
        raise ValueError("At least 4 valid 3D-2D point correspondences are required for pose estimation.")

    success, rvec, tvec = cv2.solvePnP(
        objp_world, imgp, rgb_intrinsic_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise ValueError("cv2.solvePnP failed to estimate the pose.")
    return rvec, tvec


def draw_axes_on_image(img, rgb_intrinsic_matrix, dist_coeffs, rvec, tvec, axis_length=0.2):
    """
    Draw X, Y, Z axes on the image.
      X in red
      Y in green
      Z in blue
    """
    axis = np.float32([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ]).reshape(-1, 3)

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    origin_pt, _ = cv2.projectPoints(
        np.array([[0, 0, 0]], dtype=np.float32),
        rvec, tvec, rgb_intrinsic_matrix, dist_coeffs
    )
    origin_pt = tuple(origin_pt.reshape(-1, 2).astype(int)[0])

    cv2.line(img, origin_pt, tuple(imgpts[0]), (0, 0, 255), 3)   # X - red
    cv2.line(img, origin_pt, tuple(imgpts[1]), (0, 255, 0), 3)   # Y - green
    cv2.line(img, origin_pt, tuple(imgpts[2]), (255, 0, 0), 3)   # Z - blue


def draw_correspondence_on_image(objp_world, imgp, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs, img):
    """
    Draw green circles for the detected chessboard corners (imgp),
    red circles for the reprojected points, and a line connecting them.
    """
    imgpoints2, _ = cv2.projectPoints(objp_world, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpoints2 = imgpoints2.reshape(-1, 2).astype(int)

    for pt1, pt2 in zip(imgp, imgpoints2):
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2)
        cv2.circle(img, pt1, 5, (0, 255, 0), -1)  # detected corner - green
        cv2.circle(img, pt2, 3, (0, 0, 255), -1)  # projected corner - red
        cv2.line(img, pt1, pt2, (255, 0, 0), 1)


def compute_reprojection_error(objp_world, imgp, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs):
    """
    Compute the mean reprojection error.
    """
    imgpoints2, _ = cv2.projectPoints(objp_world, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpoints2 = imgpoints2.reshape(-1, 2).astype(np.float32)
    imgp = imgp.astype(np.float32)
    error = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    return error


# -------------------------------------
# tkinter GUI main class
# -------------------------------------
class RealSenseChessboardTkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Chessboard ")

        # Chessboard parameters
        self.checkerboard_size = (6, 4)   # (cols, rows)
        self.square_size = 0.08
        self.objp_world = generate_object_points(self.checkerboard_size, self.square_size)

        # Folders to store outputs
        self.save_dir_chessboard = './single_chessboard_image'
        self.save_dir_data = './data'
        os.makedirs(self.save_dir_chessboard, exist_ok=True)
        os.makedirs(self.save_dir_data, exist_ok=True)

        # State variables
        self.pause_frame = False
        self.detected_corners = None
        self.detected_depth = None
        self.detected_color_image = None
        self.rvec = None
        self.tvec = None

        # GUI widgets
        self.image_label = tk.Label(self.root, text="Waiting for image...")
        self.image_label.pack()

        # Checkbox for 180° flip
        self.flip_var = tk.BooleanVar(value=False)
        self.flip_checkbox = tk.Checkbutton(self.root, text="Flip 180°", variable=self.flip_var)
        self.flip_checkbox.pack()

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        self.save_button = tk.Button(btn_frame, text="Save", command=self.on_save)
        self.save_button.grid(row=0, column=0, padx=5, pady=5)

        self.continue_button = tk.Button(btn_frame, text="Continue", command=self.on_continue)
        self.continue_button.grid(row=0, column=1, padx=5, pady=5)

        self.quit_button = tk.Button(btn_frame, text="Quit", command=self.on_quit)
        self.quit_button.grid(row=0, column=2, padx=5, pady=5)

        # Initialize RealSense
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.rgb_intrinsic_matrix = None
        self.distortion_coeffs = None
        self.init_realsense()

        # Periodic update of frames
        self.update_frame()

        # Closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

    def init_realsense(self):
        """Set up the RealSense pipeline and retrieve camera parameters."""
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        profile = pipeline.start(config)
        self.pipeline = pipeline

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale} meters/unit")

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        self.rgb_intrinsic_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.distortion_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)

        print("RGB Camera Intrinsic Matrix:\n", self.rgb_intrinsic_matrix)
        print("Distortion Coefficients:\n", self.distortion_coeffs)

        self.align = rs.align(rs.stream.color)

    def update_frame(self):
        """
        Periodically fetch frames from RealSense and update the tkinter UI.
        If pause_frame is True, only display the last detected chessboard image.
        """
        if self.pause_frame:
            if self.detected_color_image is not None:
                self.display_image(self.detected_color_image)
            # Schedule next update
            self.root.after(30, self.update_frame)
            return

        # Fetch frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            self.root.after(30, self.update_frame)
            return

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Perform a 180° flip if the checkbox is selected
        if self.flip_var.get():
            color_image = cv2.rotate(color_image, cv2.ROTATE_180)
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

        # Detect chessboard
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(color_image, self.checkerboard_size, corners_refined, True)

            # Store detection results
            self.detected_corners = corners_refined
            self.detected_depth = depth_image
            self.detected_color_image = color_image.copy()

            # Pause frame updates
            self.pause_frame = True
            print("[INFO] Chessboard detected. Pausing frame updates...")

        # Display the current color image in tkinter
        self.display_image(color_image)

        # Schedule next update (30 ms)
        self.root.after(30, self.update_frame)

    def display_image(self, img_bgr):
        """Convert an OpenCV BGR image to a Tkinter PhotoImage and display in a Label."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img  # Keep a reference to avoid garbage collection

    def on_save(self):
        """Save detected chessboard data and compute pose if corners are found."""
        if self.detected_corners is None:
            messagebox.showwarning("Warning", "No chessboard detected, cannot save!")
            return

        # 1) Save the image, depth, corners
        color_save_path = os.path.join(self.save_dir_chessboard, "chessboard_rgb.png")
        depth_save_path = os.path.join(self.save_dir_chessboard, "chessboard_depth.npy")
        corners_save_path = os.path.join(self.save_dir_chessboard, "chessboard_corners.npy")

        cv2.imwrite(color_save_path, self.detected_color_image)
        np.save(depth_save_path, self.detected_depth)
        np.save(corners_save_path, self.detected_corners)

        # 2) solvePnP
        imgp = self.detected_corners.reshape(-1, 2).astype(np.float32)
        try:
            self.rvec, self.tvec = estimate_pose(
                self.objp_world, imgp, self.rgb_intrinsic_matrix, self.distortion_coeffs
            )
            print("[INFO] Pose estimation succeeded.")
            print("rvec:\n", self.rvec)
            print("tvec:\n", self.tvec)

            # 3) Compute reprojection error
            error = compute_reprojection_error(
                self.objp_world, imgp, self.rvec, self.tvec,
                self.rgb_intrinsic_matrix, self.distortion_coeffs
            )
            print(f"[INFO] Mean Reprojection Error: {error}")

            # 4) Draw correspondences and axes
            result_img = self.detected_color_image.copy()
            draw_correspondence_on_image(
                self.objp_world, imgp, self.rvec, self.tvec,
                self.rgb_intrinsic_matrix, self.distortion_coeffs,
                result_img
            )
            draw_axes_on_image(
                result_img, self.rgb_intrinsic_matrix,
                self.distortion_coeffs, self.rvec, self.tvec
            )

            # Update the tkinter label with the annotated image
            self.display_image(result_img)

        except ValueError as e:
            messagebox.showwarning("Error", f"Pose Estimation Failed: {e}")
            return

        # 5) Save rvec, tvec, and intrinsics
        rvec_save_path = os.path.join(self.save_dir_data, "rotation_vectors.npy")
        tvec_save_path = os.path.join(self.save_dir_data, "translation_vectors.npy")
        intrinsic_save_path = os.path.join(self.save_dir_data, "rgb_intrinsic_matrix.npy")

        np.save(rvec_save_path, self.rvec)
        np.save(tvec_save_path, self.tvec)
        np.save(intrinsic_save_path, self.rgb_intrinsic_matrix)

        messagebox.showinfo(
            "Success",
            f"Data saved to:\n"
            f"1) {self.save_dir_chessboard}\n"
            f"   (chessboard_rgb.png, chessboard_depth.npy, chessboard_corners.npy)\n"
            f"2) {self.save_dir_data}\n"
            f"   (rotation_vectors.npy, translation_vectors.npy, rgb_intrinsic_matrix.npy)\n"
            f"Mean Reprojection Error: {error:.4f}"
        )

    def on_continue(self):
        """Resume frame updates by clearing the current detection and setting pause to False."""
        self.pause_frame = False
        self.detected_corners = None
        self.detected_depth = None
        self.detected_color_image = None
        self.rvec = None
        self.tvec = None
        print("[INFO] Continue detection...")

    def on_quit(self):
        """Stop the pipeline and close the tkinter window."""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.pipeline.stop()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = RealSenseChessboardTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
