import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageOps  # ImageOps is used to correct image orientation based on EXIF data
import tkinter as tk
import matplotlib.colors as mcolors  # Import for creating a custom colormap

#########################
# A) Determine if we are running in a PyInstaller bundled environment
#########################
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(__file__)

#########################
# B) Construct file and folder paths
#########################
xlsx_path = os.path.join(BASE_PATH, 'log', 'intersection_log_1.xlsm')
image_folder = os.path.join(BASE_PATH, 'images')
output_folder = os.path.join(BASE_PATH, 'heatmaps_with_bg')
os.makedirs(output_folder, exist_ok=True)
allowed_extensions = ['.png', '.jpg', '.jpeg']

#########################
# C) Read the Excel file
#########################
if not os.path.exists(xlsx_path):
    raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
df = pd.read_excel(xlsx_path)

#########################
# D) Define wall (plane) information
#########################
CUBES_INFO = [
    {
        "name": "Plane1",
        "color": "cyan",
        "face_points": [
            [4, 7, -5],
            [10, 7, -5],
            [4, 7, -4]
        ],
        "depth_point": [0, 7.5, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 0
    },
    {
        "name": "Plane2",
        "color": "red",
        "face_points": [
            [-1, 5, -1],
            [4, 5, -1],
            [-1, 5, 3]
        ],
        "depth_point": [0, 6, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 0
    },
    {
        "name": "Plane3",
        "color": "green",
        "face_points": [
            [-4.5, 5, -2],
            [-3, 5, -2],
            [-4.5, 5, 3]
        ],
        "depth_point": [0, 13, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 0
    },
    {
        "name": "Plane4",
        "color": "red",
        "face_points": [
            [-6, -2.5, -5],
            [-6, 18.5, -5],
            [-6, -2.5, 3]
        ],
        "depth_point": [-6.5, 0, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 1
    },
    {
        "name": "Plane5",
        "color": "red",
        "face_points": [
            [-6, 18.5, -5],
            [8, 18.5, -5],
            [-6, 18.5, 3]
        ],
        "depth_point": [0, 19, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 0
    },
    {
        "name": "Plane6",
        "color": "red",
        "face_points": [
            [5.5, 5, -0.5],
            [8, 5, -0.5],
            [5.5, 5, 3]
        ],
        "depth_point": [0, 13, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 0
    },
    {
        "name": "Plane7",
        "color": "green",
        "face_points": [
            [10, -2.5, -5],
            [10, 18, -5],
            [10, -2.5, 3]
        ],
        "depth_point": [10.5, 0, 0],
        "invert_x": True,
        "invert_y": False,
        "record_face": 1
    }
]

#########################
# E) Generate Heatmaps
#########################

# Create a custom colormap from green to red.
green_to_red = mcolors.LinearSegmentedColormap.from_list("GreenRed", ["green", "red"])

for cube_info in CUBES_INFO:
    cube_name = cube_info["name"]
    print(f"[DEBUG] Processing {cube_name} ...")

    # Filter data for the current plane
    group_data = df[df['Plane_Name'] == cube_name]
    U = group_data['v_local'].values if not group_data.empty else np.array([])
    V = group_data['u_local'].values if not group_data.empty else np.array([])

    # Calculate the geometric dimensions of the wall
    face_points = np.array(cube_info["face_points"])
    p0, p1, p2 = face_points[0], face_points[1], face_points[2]
    u_length = np.linalg.norm(p1 - p0)
    v_length = np.linalg.norm(p2 - p0)

    # Attempt to load the corresponding background image
    bg_image = None
    for ext in allowed_extensions:
        candidate_path = os.path.join(image_folder, f"{cube_name}{ext}")
        if os.path.exists(candidate_path):
            try:
                bg_image = Image.open(candidate_path)
                # Correct image orientation based on EXIF data
                bg_image = ImageOps.exif_transpose(bg_image)
                print(f"[DEBUG] Successfully loaded background image: {candidate_path}")
                orig_w, orig_h = bg_image.size
                desired_ratio = (u_length / v_length) if v_length != 0 else 1
                current_ratio = (orig_w / orig_h) if orig_h != 0 else 1
                if abs(desired_ratio - current_ratio) > 0.01:
                    new_w = int(orig_h * desired_ratio)
                    bg_image = bg_image.resize((new_w, orig_h), Image.LANCZOS)
                break
            except Exception as e:
                print(f"[ERROR] Failed to load image: {e}")
                bg_image = None
                continue

    # Dynamically adjust the figure size:
    scale = 0.5  # Each unit length is converted to 0.5 inches
    min_fig_width, min_fig_height = 6, 6
    fig_width = max(min_fig_width, u_length * scale)
    fig_height = max(min_fig_height, v_length * scale)

    # Create the figure and main axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(right=0.8)  # Reserve space on the right for the colorbar

    if bg_image is not None:
        # Use origin='upper' to avoid a 180-degree rotation
        ax.imshow(bg_image, extent=[0, u_length, 0, v_length],
                  origin='upper', aspect='equal')

    # Determine the plotting method based on the number of data points
    if len(U) == 0:
        ax.set_title(f"{cube_name} No data")
        print(f"[INFO] {cube_name}: no data")
    elif len(U) == 1:
        ax.scatter(U, V, s=30, c='blue', alpha=0.7)
        ax.set_title(f"{cube_name} Single point")
        print(f"[INFO] {cube_name}: only 1 data point")
    else:
        sns.kdeplot(x=U, y=V, fill=True, cmap=green_to_red,
                    bw_adjust=0.5, levels=50, alpha=0.5, ax=ax)
        ax.set_title(f"{cube_name} Heatmap")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, u_length])
    ax.set_ylim([0, v_length])
    ax.set_xlabel('U-axis (v_local)')
    ax.set_ylabel('V-axis (u_local)')

    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=green_to_red)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Density')

    # Determine the filename based on the data
    if len(U) == 0:
        save_filename = f"{cube_name}_no_data_heatmap_with_bg.png"
    else:
        save_filename = f"{cube_name}_heatmap_with_bg.png"
    save_path = os.path.join(output_folder, save_filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Heatmap saved at: {save_path}")

#########################
# F) Use Tkinter and Canvas to Automatically Resize Images to Fit the Window
#########################
def show_images_with_auto_resize(folder, show_no_data=False):
    """
    Display images using a Canvas. The image will automatically resize to fill the window when its size changes.
    :param folder: The folder path containing the images
    :param show_no_data: Whether to display images with "no_data" in their filename
    """
    if show_no_data:
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "no_data" not in f]
    files.sort()

    for name in files:
        path = os.path.join(folder, name)
        try:
            img_orig = Image.open(path)
            img_orig = ImageOps.exif_transpose(img_orig)
        except Exception as e:
            print(f"[ERROR] Unable to open image {path}: {e}")
            continue

        win = tk.Toplevel()
        win.title(name)
        canvas = tk.Canvas(win, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True)

        def resize_image(event, img_orig=img_orig, canvas=canvas):
            new_width = event.width
            new_height = event.height
            try:
                resized_img = img_orig.resize((new_width, new_height), Image.LANCZOS)
            except Exception as ex:
                print(f"[ERROR] Failed to resize image: {ex}")
                return
            tk_img = ImageTk.PhotoImage(resized_img)
            canvas.image = tk_img
            canvas.delete("all")
            canvas.create_image(new_width // 2, new_height // 2, image=tk_img)

        win.bind("<Configure>", resize_image)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes("-alpha", 0)

    show_images_with_auto_resize(output_folder, show_no_data=False)

    root.mainloop()
