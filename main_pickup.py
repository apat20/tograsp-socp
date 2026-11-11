
import argparse, os
import sys

import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from func.utils import plot_reference_frames

from pickup_runner import run_pickup


def build_cloud_object(cloud_object, pcd):
    '''
    Function builds an object of the point_cloud class

    Args:
        cloud_object - Object of class point_cloud
        pcd - Open3D point cloud object created 

    Returns:
        cloud_object
    '''
    cloud_object.processed_cloud = pcd
    cloud_object.points = np.asarray(cloud_object.processed_cloud.points)

    # Computing the normals for this point cloud:
    cloud_object.processed_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    cloud_object.processed_cloud.estimate_normals()
    cloud_object.processed_cloud.orient_normals_consistent_tangent_plane(30)
    cloud_object.normals_base_frame = np.asarray(cloud_object.processed_cloud.normals)
    return cloud_object


def get_transformed_cloud(points, g):
    '''
    Function to transform a point cloud given a rotation matrix and a position vector

    Args:
        points:
        g:

    Returns:
        transformed_points:

    '''
    transformed_points = np.asarray([np.reshape(np.add(np.matmul(g[0:3, 0:3], point), g[0:3, 3]), [1,3]) for point in points])
    return np.reshape(transformed_points, [points.shape[0], points.shape[1]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="nontextured.ply")
    parser.add_argument("--trial", default="1")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    out_dir = os.path.join(args.log_dir, f"pickup_trial_{args.trial}")
    os.makedirs(out_dir, exist_ok=True)

    # Check if the file exists
    if not os.path.isfile(args.filename):
        sys.exit(f"ERROR: The file {args.filename} does not exist. Exiting script")
 
    # Reading the point cloud from a file
    pcd = o3d.io.read_point_cloud(args.filename)

    # Computing grasping region and corresponding poses for picking the object up (pure translation)
    grasp_object, cloud_object = run_pickup(pcd, args.trial, args.log_dir)

    # VISUALIZATION:
    x_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 0], [cloud_object.transformed_points_object_frame.shape[0],1])
    y_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 1], [cloud_object.transformed_points_object_frame.shape[0],1])
    z_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 2], [cloud_object.transformed_points_object_frame.shape[0],1])

    '''PLOT 1:'''
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.grid(False)

    # # Plot the object bounding box and point cloud:
    grasp_object.vertices = cloud_object.transformed_vertices_object_frame
    grasp_object.plot_cube()
    ax1.add_collection3d(Poly3DCollection(grasp_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    ax1.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)

    # # Visualize the screw axis: 
    # ax1.scatter(grasp.point_base[0], grasp.point_base[1], grasp.point_base[2], marker = '*', s = 100, color = 'r')
    # ax1.quiver(grasp.point_base[0], grasp.point_base[1], grasp.point_base[2], 0.25*grasp.unit_vector_base[0], 0.25*grasp.unit_vector_base[1], 0.25*grasp.unit_vector_base[2], color = "r", arrow_length_ratio = 0.25)

    # Base configuration:
    g_base = np.eye(4)
    ax1 = plot_reference_frames(g_base[0:3, 0:3], np.reshape(g_base[0:3, 3], [3]), 0.08, 0.08, ax1)

    for pose in grasp_object.computed_end_effector_poses:
        ax1 = plot_reference_frames(pose[0:3, 0:3], np.reshape(pose[0:3, 3], [3]), 0.02, 0.02, ax1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    '''PLOT 2:'''
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.grid(False)

    # # Plot the object bounding box and point cloud:
    grasp_object.vertices = cloud_object.transformed_vertices_object_frame
    grasp_object.plot_cube()
    ax2.add_collection3d(Poly3DCollection(grasp_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    ax2.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)

    # Base configuration:
    ax2 = plot_reference_frames(g_base[0:3, 0:3], np.reshape(g_base[0:3, 3], [3]), 0.08, 0.08, ax2)

    for gc in grasp_object.grasp_centers:
        ax2.scatter(gc[0], gc[1], gc[2], marker = '*', s = 100, color = 'r')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)

    plt.show()