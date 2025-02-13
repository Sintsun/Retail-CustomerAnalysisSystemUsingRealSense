import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants for 3D plot limits and cube data
X_LIMITS = (-10, 10)
Y_LIMITS = (-16, 16)
Z_LIMITS = (-10, 3)

CUBES_INFO = [
    {
        "name": "Plane1",
        "color": "red",
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
        "color": "blue",
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
        "color": "yellow",
        "face_points": [
            [-6.5, -2.5, -5],
            [-6.5, 18.5, -5],
            [-6.5, -2.5, 3]
        ],
        "depth_point": [-6, 0, 0],
        "invert_x": True,
        "invert_y": True,
        "record_face": 1
    },
    {
        "name": "Plane5",
        "color": "purple",
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
        "color": "brown",
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
        "color": "black",
        "face_points": [
            [10.5, -2.5, -5],
            [10.5, 18, -5],
            [10.5, -2.5, 3]
        ],
        "depth_point": [10, 0, 0],
        "invert_x": True,
        "invert_y": False,
        "record_face": 1
    }
    # ,
    # {
    #     "name": "Wall1",
    #     "color": "cyan",
    #     "face_points": [
    #         [-1.75, 3, -3],
    #         [1.75, 3, -3],
    #         [-1.75, 3, 0]
    #     ],
    #     "depth_point": [0, 5, 0],
    #     "invert_x": True,
    #     "invert_y": True,
    #     "record_face": 0
    # },
    # {
    #     "name": "Wall2",
    #     "color": "red",
    #     "face_points": [
    #         [-2, -3, -3],
    #         [-2, 3, -3],
    #         [-2, -3, 0]
    #     ],
    #     "depth_point": [-1.75, 0, 0],
    #     "invert_x": True,
    #     "invert_y": True,
    #     "record_face": 1
    # },
#     {
#         "name": "Wall3",
#         "color": "green",
#         "face_points": [
#             [2, -3, -3],
#             [2, 3, -3],
#             [2, -3, 0]
#         ],
#         "depth_point": [1.75, 5, 0],
#         "invert_x": True,
#         "invert_y": False,
#         "record_face": 1
#     }

 ]

def plot_cube_with_index(ax, face_points, depth_point, color='r', alpha=0.5, plane_name=""):
    # 將輸入的點轉成 numpy 陣列以便向量計算
    p0, p1, p2 = face_points
    vec1 = p1 - p0
    vec2 = p2 - p0

    if np.allclose(np.cross(vec1, vec2), 0):
        raise ValueError("Invalid face points (collinear).")

    # 根據輸入的三個點計算第四個點
    p3 = p1 + (p2 - p0)
    front_face = [p0, p1, p3, p2]

    # 計算面法向量以及深度偏移
    normal = np.cross(vec1, vec2)
    normal /= np.linalg.norm(normal)
    dp_vec = depth_point - p0
    depth = np.dot(dp_vec, normal)
    offset = depth * normal

    # 根據偏移計算後面的面
    back_face = [v + offset for v in front_face]

    # 定義立方體側面
    side1 = [front_face[0], front_face[1], back_face[1], back_face[0]]
    side2 = [front_face[1], front_face[2], back_face[2], back_face[1]]
    side3 = [front_face[2], front_face[3], back_face[3], back_face[2]]
    side4 = [front_face[3], front_face[0], back_face[0], back_face[3]]
    cube_faces = []
    cube_faces.append((0, front_face))
    cube_faces.append((1, back_face))
    cube_faces.append((2, side1))
    cube_faces.append((3, side2))
    cube_faces.append((4, side3))
    cube_faces.append((5, side4))

    # 將多邊形加入到 3D 圖中
    if ax is not None:
        poly3d_list = [face for idx, face in cube_faces]
        poly = Poly3DCollection(poly3d_list, facecolors=color, edgecolors='k', alpha=alpha)
        ax.add_collection3d(poly)

        # 只繪製 face_points 中的三個頂點，並給予指定顏色
        vertices = np.array([p0, p1, p2])
        vertex_colors = ['red', 'green', 'blue']  # 可以根據需要更改顏色
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vertex_colors, s=50)

        # 在前面板的中心標註平面的名稱
        centroid = np.mean(front_face, axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], plane_name, color='black', fontsize=12)

    return cube_faces

def test_3d_plot():
    # Create the figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp')

    # Set axis limits
    ax.set_xlim(X_LIMITS)
    ax.set_ylim(Y_LIMITS)
    ax.set_zlim(Z_LIMITS)

    # Invert the x-axis to swap left and right
    ax.invert_xaxis()

    # Invert the z-axis so that the negative z-direction is at the top
    ax.invert_zaxis()

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot title to "Shop 3D Model Test"
    ax.set_title('Shop 3D Model Test')

    # Add cubes to the plot
    legend_patches = []
    for cube in CUBES_INFO:
        face_points = np.array(cube["face_points"], dtype=float)
        depth_point = np.array(cube["depth_point"], dtype=float)
        plot_cube_with_index(ax, face_points, depth_point, color=cube["color"], alpha=0.6, plane_name=cube["name"])

        # Add legend entry
        legend_patches.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=cube["color"], markersize=10,
                                         label=cube["name"]))

    # Add legend to the plot
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Show the plot
    plt.show()

# Run the test function
if __name__ == "__main__":
    test_3d_plot()
