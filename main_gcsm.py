
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math

# Importing quaternion functionalities
import func.quaternion_lib as ql

# Import helper functions for visualisation
from func.utils import plot_cube
from func.utils import plot_reference_frames

# Functionalities for point cloud processing and computing the ideal grasping region:
from point_cloud_module.process_point_cloud import point_cloud
from point_cloud_module.tograsp_socp import tograsp

# Second Order Cone Program for computing the task-depedent grasp metric:
from socp_module.two_finger_gfop import two_finger_gfop


def build_cloud_object(cloud_object, pcd):
    '''
    Function builds an object of the point_cloud class

    Args:


    Returns:
    
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

    Returns:

    '''
    transformed_points = np.asarray([np.reshape(np.add(np.matmul(g[0:3, 0:3], point), g[0:3, 3]), [1,3]) for point in points])
    return np.reshape(transformed_points, [points.shape[0], points.shape[1]])

def transform_screw_axis(unit_vector, point, g):
    '''
    Function to transform the unit vector and screw axis and express it with respect to the desired pose/configuratio 'g'

    Args:

    Returns:

    '''
    u = np.asarray([unit_vector[0], unit_vector[1], unit_vector[2], 0])
    p = np.asarray([point[0], point[1], point[2], 1])
    unit_vector_transformed = g @ u.T
    point_transformed = g @ p.T
    return unit_vector_transformed, point_transformed

def build_and_solve_gfop(gfop_object, grasp_object):
    '''
    Function to build and solve an instance of the Grasping Force Optimization(gfop) class

    Args:

    Returns:

    '''
    # Data for formalizing the optimization problem for computing the task dependent grasp metric:
    # Initial Object Configuration
    gfop_object.R_O_initial = np.eye(3, dtype=float)
    gfop_object.p_O_initial = np.zeros([3,1])

    # Final Object Configuration
    gfop_object.R_O_final = np.zeros([3,3])
    gfop_object.p_O_final = np.zeros([3,1])

    if grasp_object.cloud_object.y_dim < grasp_object.gripper_width_tolerance:
        print('Generating contacts along XZ plane')
        # Object-Robot Contact location position and orientation information with respect to the Object reference frame:
        # Contact location C1:
        gfop_object.R_OC_1 = np.asarray([[1,0,0],[0,0,1],[0,-1,0]])
        gfop_object.normal_C_1 = gfop_object.R_OC_1[0:3, 2]
        # Contact location C2:
        gfop_object.R_OC_2 = np.asarray([[-1,0,0],[0,0,-1],[0,-1,0]])
        gfop_object.normal_C_2 = gfop_object.R_OC_2[0:3, 2]
    elif grasp_object.cloud_object.x_dim < grasp_object.gripper_width_tolerance:
        print('Generating contacts along YZ plane')
        # Object-Robot Contact location position and orientation information with respect to the Object reference frame:
        gfop_object.R_OC_1 = np.asarray([[0,0,1],[0,1,0],[-1,0,0]])
        # Contact normal at contact location C1:
        gfop_object.normal_C_1 = gfop_object.R_OC_1[0:3, 2]
        gfop_object.R_OC_2 = np.asarray([[0,0,-1],[0,-1,0],[-1,0,0]])
        # Contact normal at contact location C2:
        gfop_object.normal_C_2 = gfop_object.R_OC_2[0:3, 2]
    elif grasp_object.cloud_object.x_dim < grasp_object.gripper_width_tolerance and grasp_object.y_dim < grasp_object.gripper_width_tolerance:
        print('Both dimensions with gripper width tolerance. Generating contacts along XZ plane')

    # Maximum force in Newtons that the fingers can apply in the normal direction:
    gfop_object.F = 30

    # Specifying the contact model to compute the Grasp Map. Here the contact model is the soft finger contact model
    gfop_object.contact_model = 'sf'

    # External force acting on the object. For this application we assume that the external force is the self weight of 
    # the object.
    # NOTE: Dimension of the array if 1x6 while initializing. It should be 6x1 when solving the optimization problem. 
    gfop_object.F_external = np.reshape(np.asarray([0,0,-10,0,0,0]), [6,1])

    # Variables specifying the dimensions of the optimization variables in the SOCP:
    gfop_object.m = 6
    gfop_object.n = 1

    # Friction parameters for soft finger contact at the object-robot contact locations:
    # Contact location 1:
    gfop_object.e11 = 1
    gfop_object.e12 = 1
    gfop_object.e1r = 0.2
    gfop_object.mu1 = 0.4

    # Contact location 2:
    gfop_object.e21 = 1
    gfop_object.e22 = 1
    gfop_object.e2r = 0.2
    gfop_object.mu2 = 0.4

    gfop_object.sigma  = 0.6

    # Friction coefficient for point contact with friction model assumed at the object-environment contact:
    gfop_object.mu = 0.6
    
    computed_metric_values = []

    # Computing the metric for all the sampled contact locations:
    for i in range(grasp_object.sampled_c1.shape[0]):
        gfop_object.p_OC_1 = np.reshape(grasp_object.sampled_c1[i], [3,1])
        gfop_object.p_OC_2 = np.reshape(grasp_object.sampled_c2[i], [3,1])
        # metric = gfop.computeMetric()
        metric = gfop_object.compute_metric_moment()
        computed_metric_values.append(metric)
    return computed_metric_values


def get_logs(grasp_object, dir, trial):
    '''
    Function to save all the necessary log files

    Args:

    Returns: None
    '''

    # Saving the initial point cloud expressed in the local object reference frame: 
    object_frame_cloud = o3d.geometry.PointCloud()
    object_frame_cloud.points = o3d.utility.Vector3dVector(grasp_object.cloud_object.transformed_points_object_frame.astype(np.float64))
    object_frame_cloud.paint_uniform_color([0, 0, 1])
    o3d.io.write_point_cloud(f"{dir}/pickup_trial_{trial}/initial_cloud_object_frame.ply", object_frame_cloud)

    # Saving the projected point cloud expressed in the object reference frame:
    # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
    object_frame_projected_cloud = o3d.geometry.PointCloud()
    object_frame_projected_cloud.points = o3d.utility.Vector3dVector(grasp_object.projected_points.astype(np.float64))
    object_frame_projected_cloud.paint_uniform_color([0, 0, 1])
    o3d.io.write_point_cloud(f"{dir}/pickup_trial_{trial}/projected_object_frame.ply", object_frame_projected_cloud)

    # Saving the unit vector and the point corresponding to the screw axis:
    np.savetxt(f"{dir}/pickup_trial_{trial}/unit_vector.csv", grasp_object.unit_vector)
    np.savetxt(f"{dir}/pickup_trial_{trial}/point.csv", grasp_object.point)

    # Saving the transformed vertices so that they can be loaded for metric computation: 
    np.savetxt(f"{dir}/pickup_trial_{trial}/transformed_vertices_object_frame.csv", grasp_object.cloud_object.transformed_vertices_object_frame, delimiter=',')

    # Saving cloud_object.x_data for verification:
    np.savetxt(f"{dir}/pickup_trial_{trial}/x_data.csv", grasp_object.x_data, delimiter=',')

    # Saving cloud_object.test_datapoints and cloud_object.predicted for post-processing:
    np.savetxt(f"{dir}/pickup_trial_{trial}/test_datapoints.csv", grasp_object.test_datapoints, delimiter=',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/test_predicted.csv", grasp_object.computed, delimiter=',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/test_metric_values.csv", grasp_object.metric_values, delimiter=',')

    # Saving the computed metric grid and corresponding parameters:
    np.savetxt(f"{dir}/pickup_trial_{trial}/metric_grid_computed.csv", grasp_object.metric_grid, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/metric_grid_occupied_computed.csv", grasp_object.grid_metric_values_occupied, delimiter = ',')

    np.savetxt(f"{dir}/pickup_trial_{trial}/X_grid_points_computed.csv", grasp_object.X_grid_points, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/Y_grid_points_computed.csv", grasp_object.Y_grid_points, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/Z_grid_points_computed.csv", grasp_object.Z_grid_points, delimiter = ',')

    np.savetxt(f"{dir}/pickup_trial_{trial}/X_grid_points_occupied_computed.csv", grasp_object.X_grid_points_occupied, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/Y_grid_points_occupied_computed.csv", grasp_object.Y_grid_points_occupied, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/Z_grid_points_occupied_computed.csv", grasp_object.Z_grid_points_occupied, delimiter = ',')

    np.savetxt(f"{dir}/pickup_trial_{trial}/q_x_array_computed.csv", grasp_object.q_x_array, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/q_y_array_computed.csv", grasp_object.q_y_array, delimiter = ',')
    np.savetxt(f"{dir}/pickup_trial_{trial}/q_z_array_computed.csv", grasp_object.q_z_array, delimiter = ',')


if __name__ == "__main__":

    # Read and process the point cloud:
    filename = "nontextured.ply"

    # Filepath to save logs for visualization and debugging:
    log_dir = 'logs/'
    trial = '2'

    # Creating the cloud object and loading the necessary file:
    cloud = point_cloud()   
    pcd = o3d.io.read_point_cloud(filename)
    cloud = build_cloud_object(cloud, pcd)

    # Computing the bounding boxes corresponding to the object point cloud: 
    cloud.compute_bounding_box()

    # Initialize an object of tograsp:
    grasp = tograsp(cloud)

    # Specifying gripper tolerances:
    grasp.gripper_width_tolerance = 0.08
    # Original tips:
    grasp.gripper_height_tolerance = 0.041
    # Attributes to compute the location of the reference frame at the flange for the grasp pose and pre-grasp pose
    grasp.g_delta = 0.0624
    grasp.g_delta_inter = 0.0925

    # Base/world reference frame:
    g_base = np.eye(4)

    # TASK-SPACE PATH: 
    # Intermediate pose 1:
    # Rotation about X-axis:
    # theta = math.radians(30)
    theta_X = 30
    R_theta_X = np.asarray([[1, 0, 0],
                            [0, np.cos(theta_X), -np.sin(theta_X)],
                            [0, np.sin(theta_X), np.cos(theta_X)]])
    theta_Y = 10
    R_theta_Y = np.asarray([[np.cos(theta_Y), 0, np.sin(theta_Y)],
                            [0, 1, 0],
                            [-np.sin(theta_Y), 0, np.cos(theta_Y)]])
    
    # R_inter_1, p_inter_1, g_inter_1 = R_theta_X @ np.eye(3), np.asarray([0, 0.15, 0.4]), np.eye(4)
    R_inter_1, p_inter_1, g_inter_1 = R_theta_Y @ R_theta_X @ np.eye(3), np.asarray([0.2, 0.15, 0.4]), np.eye(4)

    g_inter_1[0:3, 0:3], g_inter_1[0:3, 3] = R_inter_1, p_inter_1

    # SCREW LINEAR INTERPOLATION TO COMPUTE THE TASK SPACE PATH:
    # Computing the intermediate configurations using screw linear interpolation for the first constant screw motion:
    [R_array_1, p_array_1, C_array_1, G_array_1, screw_params_1] = ql.sclerp(g_base[0:3, 0:3], g_base[0:3, 3], 
                                                          R_inter_1, p_inter_1)
    
    # Extracting the screw parameters:
    # NOTE: The screw parameters are always computed with respect to the initial configuration given as input to ScLERP.
    theta_1 = screw_params_1[0]
    point_1 = screw_params_1[1]
    unit_vector_1 = screw_params_1[2]
    pitch_1 = screw_params_1[3]
    m_1 = screw_params_1[4]

    print(f"Screw Parameters for the first constant screw motion:\n {screw_params_1}")
    print(f"Unit vector: \n{unit_vector_1}")

    grasp.unit_vector_base, grasp.point_base = transform_screw_axis(unit_vector_1, point_1, g_base)

    # COMPUTING THE TASK-DEPENDENT GRASP METRIC AS A SOCP:
    # Initializing object of the GFOP class:
    gfop = two_finger_gfop()

    # Generating antipodal contact locations along two parallel faces of the bounding box:
    grasp.generate_contacts()

    # Extracting the screw axis from the datapoints:
    gfop.unit_vector = np.reshape(grasp.unit_vector, [3,1])
    
    # Build and solve gfop_1:
    computed_metric_values_1 = build_and_solve_gfop(gfop, grasp)
    grasp.test_datapoints = grasp.x_data
    grasp.computed = np.reshape(np.asarray(computed_metric_values_1), [len(computed_metric_values_1), 1])

    # Normalizing the values between 0 and 1:
    grasp.computed = (grasp.computed - np.min(grasp.computed))/(np.max(grasp.computed - np.min(grasp.computed)))
    grasp.metric_values = grasp.computed

    # Computing the ideal grasping region:
    grasp.get_ideal_grasping_region()

    # Computing the end-effector poses based on the ideal grasping region:
    grasp.get_end_effector_poses()

    # VISUALIZATION:
    x_transformed_points = np.reshape(cloud.transformed_points_object_frame[:, 0], [cloud.transformed_points_object_frame.shape[0],1])
    y_transformed_points = np.reshape(cloud.transformed_points_object_frame[:, 1], [cloud.transformed_points_object_frame.shape[0],1])
    z_transformed_points = np.reshape(cloud.transformed_points_object_frame[:, 2], [cloud.transformed_points_object_frame.shape[0],1])

    # Saving logs for visualization:
    get_logs(grasp, log_dir, trial)

    '''PLOT 1:'''
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.grid(False)

    # # Plot the object bounding box and point cloud:
    grasp.vertices = cloud.transformed_vertices_object_frame
    grasp.plot_cube()
    ax1.add_collection3d(Poly3DCollection(grasp.faces, linewidths=1, edgecolors='b', alpha=.25))
    # ax1.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)

    # # Visualize the screw axis (base reference frame): 
    # ax1.scatter(grasp.point_base[0], grasp.point_base[1], grasp.point_base[2], marker = '*', s = 100, color = 'r')
    # ax1.quiver(grasp.point_base[0], grasp.point_base[1], grasp.point_base[2], 0.25*grasp.unit_vector_base[0], 0.25*grasp.unit_vector_base[1], 0.25*grasp.unit_vector_base[2], color = "r", arrow_length_ratio = 0.25)

    # Visualize the screw axis (original reference frame): 
    ax1.scatter(point_1[0], point_1[1], point_1[2], marker = '*', s = 100, color = 'r')
    ax1.quiver(point_1[0], point_1[1], point_1[2], 0.25*unit_vector_1[0], 0.25*unit_vector_1[1], 0.25*unit_vector_1[2], color = "r", arrow_length_ratio = 0.25)

    # Base configuration:
    ax1 = plot_reference_frames(g_base[0:3, 0:3], np.reshape(g_base[0:3, 3], [3]), 0.08, 0.08, ax1)

    # Final configuration:
    ax1 = plot_reference_frames(g_inter_1[0:3, 0:3], np.reshape(g_inter_1[0:3, 3], [3]), 0.08, 0.08, ax1)

    for pose in grasp.computed_end_effector_poses:
        ax1 = plot_reference_frames(pose[0:3, 0:3], np.reshape(pose[0:3, 3], [3]), 0.02, 0.02, ax1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)

    plt.show()