#!/usr/bin/env python3
"""
Grasp synthesis to pickup a single one object, using a point cloud obtained from via a perception system.

    python3 pickup.py cup --base-frame <robot_frame from perception's system.json>

This code is similar in structure to the perception pipeline's 'detect "cup"'.
It asks perception for the object, runs
run_pickup() on that object's cloud (your bounding box, grasp metric and end-effector poses),
publishes the poses and prints a summary. Perception's own pose and size for the object are
not used. 

NOTE: multi-word names: python3 pickup.py "'tennis ball'"

Published in the robot base frame, stamped with the capture time:
    /tograsp/grasp_poses, /tograsp/pregrasp_poses   geometry_msgs/PoseArray (flange poses)
    /tograsp/object_box/pose                        geometry_msgs/PoseStamped
    /tograsp/object_box/size                        geometry_msgs/Vector3Stamped (along the box axes, m)
    /tograsp/object_box/marker                      visualization_msgs/Marker (for RViz)
    /tograsp/object_cloud                           sensor_msgs/PointCloud2 (the cloud used, for RViz)
The same poses are saved to <log-dir>/pickup_trial_<request_id>/poses.json.

NOTE: Perception's own pose and size for the object, computed using ros_seg_pipeline/perception/geometry.py are not used 

ROS 2 keeps the published poses available to subscribers that connect later only while this
script runs, so after publishing it waits for Ctrl+C. Use --hold SECONDS to exit on its own.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from func.utils import plot_reference_frames

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Vector3Stamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from seg_interfaces.action import DetectObjects
from seg_interfaces.msg import ObjectClouds

from pickup_runner import run_pickup

# HELPER FUNCTIONS:
def cloud_to_points(cloud):
    """
    N x 3 float64 array from a PointCloud2 written by perception's ros_io.cloud_msg.
    
    Args:

    Returns:
    
    """
    layout = [(f.name, f.offset) for f in cloud.fields]
    if layout != [("x", 0), ("y", 4), ("z", 8)] or cloud.point_step != 12 or cloud.is_bigendian:
        raise ValueError(f"unexpected PointCloud2 layout {layout}, point_step {cloud.point_step}")
    return np.frombuffer(cloud.data, dtype="<f4").reshape(-1, 3).astype(np.float64)


def matrix_to_pose(T):
    """
    geometry_msgs/Pose from a 4x4 homogeneous transform.
    
    Args:

    Returns:
    
    """
    T = np.asarray(T, dtype=float)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    # ROS orders quaternions (x, y, z, w), as SciPy does by default; func/quaternion_lib puts w first.
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    pose.orientation.x, pose.orientation.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = float(z), float(w)
    return pose


def pose_values(pose):
    """
    [x, y, z, qx, qy, qz, qw] for printing and saving.
    
    Args:

    Returns:
    
    """
    p, q = pose.position, pose.orientation
    return [round(v, 5) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)]


def plot_results(grasp, cloud, out_dir):
    """
    Saves PNGs to out_dir and shows the windows (depends if matplotlib has an interactive backend)

    Args:

    Returns:
        The saved file paths.
    """

    def scene(points, vertices, title):
        grasp.vertices = vertices
        grasp.plot_cube()  # fills grasp.faces from grasp.vertices
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.grid(False)
        ax.add_collection3d(Poly3DCollection(grasp.faces, linewidths=1, edgecolors="b", alpha=.25))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
        plot_reference_frames(np.eye(3), np.zeros(3), 0.08, 0.08, ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        return fig, ax

    saved = []

    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150)
        saved.append(path)

    # Figures 1 and 2: as in main_pickup.py, in the object (bounding box) frame.
    pts_obj, verts_obj = cloud.transformed_points_object_frame, cloud.transformed_vertices_object_frame
    fig, ax = scene(pts_obj, verts_obj, "Grasp poses, object frame")
    for pose in grasp.computed_end_effector_poses:
        plot_reference_frames(pose[0:3, 0:3], np.reshape(pose[0:3, 3], [3]), 0.02, 0.02, ax)
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-1, 1)
    save(fig, "grasp_poses_object_frame.png")

    fig, ax = scene(pts_obj, verts_obj, "Grasp centres, object frame")
    for gc in grasp.grasp_centers:
        ax.scatter(gc[0], gc[1], gc[2], marker="*", s=100, color="r")
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-1, 1)
    save(fig, "grasp_centres_object_frame.png")

    # Figure 3: what the robot receives. Cloud, box and poses in the robot base frame.
    g = np.asarray(cloud.g_bounding_box)
    verts_base = verts_obj @ g[:3, :3].T + g[:3, 3]
    fig, ax = scene(np.asarray(cloud.points), verts_base, "Grasp poses, robot base frame")
    for pose in grasp.computed_end_effector_poses_base:
        plot_reference_frames(pose[0:3, 0:3], np.reshape(pose[0:3, 3], [3]), 0.02, 0.02, ax)
    shown = np.vstack([cloud.points] + [p[0:3, 3] for p in grasp.computed_end_effector_poses_base])
    centre, half = shown.mean(axis=0), np.ptp(shown, axis=0).max() / 2 + 0.05
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        set_lim(c - half, c + half)
    save(fig, "grasp_poses_base_frame.png")

    if plt.get_backend().lower() != "agg":
        plt.show()  # returns when the windows are closed
    plt.close("all")
    return saved

def next_trial_number(log_dir):
    """
    Trial = highest numbered pickup_trial_<n> folder in log_dir + , or 1 if there is none.

    Args:

    Returns:

    """
    names = os.listdir(log_dir) if os.path.isdir(log_dir) else []
    used = [int(n.rsplit("_", 1)[1]) for n in names
            if n.startswith("pickup_trial_") and n.rsplit("_", 1)[1].isdigit()]
    return max(used, default=0) + 1

def select_object(result, query):
    """
    Highest-scoring object that has a pose, preferring objects whose label matches the query.
    
    Args:

    Returns:
    
    """
    wanted = query.strip().strip("'\"").lower()
    with_pose = [o for o in result.objects if o.has_pose]
    matching = [o for o in with_pose if o.label == wanted]
    pool = matching or with_pose
    return max(pool, key=lambda o: o.score) if pool else None

# ROS 2 Pickup Node
class Pickup(Node):
    def __init__(self, base_frame, log_dir, timeout, trial = None):
        super().__init__("tograsp_pickup")

        self.base_frame = base_frame
        self.log_dir = log_dir
        self.timeout = timeout
        self.trial = trial
        
        # Same QoS as perception's outputs: the latest message stays available to late subscribers.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        
        self.detect = ActionClient(self, DetectObjects, "/perception/detect")
        self.clouds = {}
        
        # Setting up the node to listen to certain topics
        self.create_subscription(ObjectClouds, "/perception/clouds", self.on_clouds, latched)
        
        # Setting up the node to publish specified topics
        self.grasp_pub = self.create_publisher(PoseArray, "/tograsp/grasp_poses", latched)
        self.pregrasp_pub = self.create_publisher(PoseArray, "/tograsp/pregrasp_poses", latched)
        self.box_pose_pub = self.create_publisher(PoseStamped, "/tograsp/object_box/pose", latched)
        self.box_size_pub = self.create_publisher(Vector3Stamped, "/tograsp/object_box/size", latched)
        self.box_marker_pub = self.create_publisher(Marker, "/tograsp/object_box/marker", latched)
        self.cloud_pub = self.create_publisher(PointCloud2, "/tograsp/object_cloud", latched)

    def on_clouds(self, msg):
        self.clouds[msg.request_id] = msg

    def run(self, query):
        """
        Runner function to query the perception system for an object, retrieve its point cloud and compute a set of grasps 
        based on the desired motion. In this particular example, the desired motion is pure translation.

        Args:


        Return:


        """
        start = time.monotonic()

        # 1. Ask perception for the object 
        # NOTE: This is essentially what `detect "cup"` does.
        if not self.detect.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("perception's /perception/detect action is not available; is it running?")
        
        # NOTE: goal_future, goal, result_future are standard ROS 2 action-client objects available via rclpy
        goal_future = self.detect.send_goal_async(
            DetectObjects.Goal(query=query),
            feedback_callback=lambda msg: print(f"perception: {msg.feedback.stage}", flush=True))
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=10.0)

        if not goal_future.done() or not goal_future.result().accepted:
            raise RuntimeError("perception did not accept the request")
        goal = goal_future.result()

        result_future = goal.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.timeout)
        if not result_future.done():
            rclpy.spin_until_future_complete(self, goal.cancel_goal_async(), timeout_sec=2.0)
            raise TimeoutError(f"no perception result within {self.timeout:g} s")
        
        result = result_future.result().result.result
        if result.status != "ok":
            raise RuntimeError(f"perception returned '{result.status}': {result.message}")

        # 2. Choose the object and take its cloud (published by perception just before the result).
        obj = select_object(result, query)
        if obj is None:
            raise RuntimeError(f"no detected object has a pose ({[o.label for o in result.objects]})")
        deadline = time.monotonic() + 5.0
        while result.request_id not in self.clouds and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        clouds = self.clouds.get(result.request_id)
        entry = clouds and next((c for c in clouds.objects if c.object_id == obj.id), None)
        if entry is None:
            raise RuntimeError(f"no cloud arrived for object {obj.id} of request {result.request_id}")
        frame = entry.cloud.header.frame_id
        if self.base_frame and frame != self.base_frame:
            raise RuntimeError(
                f"cloud is in '{frame}', not '{self.base_frame}': check robot_frame in system.json")
        points = cloud_to_points(entry.cloud)

        # 3. Bounding box, grasp metric and end-effector poses, exactly as in main_pickup.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Logging the data based on the trial number
        trial = self.trial or next_trial_number(self.log_dir)
        # trial = result.request_id
        trial_dir = os.path.join(self.log_dir, f"pickup_trial_{trial}")
        os.makedirs(trial_dir, exist_ok=True)  # get_logs writes here and does not create it
        try:
            grasp, cloud = run_pickup(pcd, trial, self.log_dir)
        except SystemExit as exc:  # run_pickup exits when no SOCP instance solves
            raise RuntimeError(str(exc)) from None

        # 4. Publish in the cloud's frame, stamped with the capture time, and save the same poses.
        header = Header(stamp=result.header.stamp, frame_id=frame)
        box_pose = matrix_to_pose(cloud.g_bounding_box)
        # Unrounded extents of the box along its own axes (x_dim, y_dim, z_dim are rounded to 1 cm).
        size = [float(v) for v in np.ptp(cloud.transformed_vertices_object_frame, axis=0)]
        grasps = [matrix_to_pose(T) for T in grasp.computed_end_effector_poses_base]
        pregrasps = [matrix_to_pose(T) for T in grasp.computed_end_effector_poses_inter_base]

        # Utilizing standard geometric message types for the messages. 
        box_size = Vector3Stamped(header=header)
        box_size.vector.x, box_size.vector.y, box_size.vector.z = size
        marker = Marker(header=header, ns="tograsp", id=0, type=Marker.CUBE,
                        action=Marker.ADD, pose=box_pose)
        marker.scale.x, marker.scale.y, marker.scale.z = size
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 0.6, 1.0, 0.4
        self.cloud_pub.publish(entry.cloud)
        self.box_pose_pub.publish(PoseStamped(header=header, pose=box_pose))
        self.box_size_pub.publish(box_size)
        self.box_marker_pub.publish(marker)
        self.pregrasp_pub.publish(PoseArray(header=header, poses=pregrasps))
        self.grasp_pub.publish(PoseArray(header=header, poses=grasps))

        # Plotting the results. For the time being the results are being saved to a directory
        # NOTE: Matplotlib will show the interactive window (possibly?)
        for path in plot_results(grasp, cloud, trial_dir):
            print(f"Figure saved: {path}", flush=True)

        # Dictionary data structure to store the output and results
        summary = {
            # "request_id": trial,
            "request_id": result.request_id,
            "trial": trial,
            "query": query,
            "object": {"id": obj.id, "label": obj.label, "score": round(obj.score, 3),
                       "points": len(points)},
            "frame": frame,
            "box_pose_xyz_xyzw": pose_values(box_pose),
            "box_size_m": [round(v, 4) for v in size],
            "grasp_poses_xyz_xyzw": [pose_values(p) for p in grasps],
            "pregrasp_poses_xyz_xyzw": [pose_values(p) for p in pregrasps],
        }

        # logging the output
        with open(os.path.join(trial_dir, "poses.json"), "w") as f:
            json.dump(summary, f, indent=1)
        summary["seconds"] = round(time.monotonic() - start, 2)
        summary["saved_to"] = os.path.join(trial_dir, "poses.json")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Pickup grasp planning on a perception point cloud")

    # Similar to how the object input is currently given to the perception system
    parser.add_argument("query", help="object to pick, e.g. cup or \"'tennis ball'\"")

    # This is not required    
    parser.add_argument("--base-frame", default="",
                        help="frame the cloud must be in: robot_frame in perception's system.json")
    
    # Directory to log the results
    parser.add_argument("--log-dir", default="logs", help="where run_pickup writes its logs")
    parser.add_argument("--trial", default=None, help="name for the log folder pickup_trial_<trial> (default: next free number)")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for perception")
    parser.add_argument("--hold", type=float, default=None,
                        help="exit this many seconds after publishing (default: wait for Ctrl+C)")
    
    # To visualize the results
    parser.add_argument("--plot", action="store_true", help="draw, save and show the matplotlib figures")
    args = parser.parse_args()

    rclpy.init()

    # Initialize the pickup node
    # node = Pickup(args.base_frame, args.log_dir, args.timeout)
    node = Pickup(args.base_frame, args.log_dir, args.timeout, trial=args.trial)
    code = 0
    try:
        s = node.run(args.query)
        b, n = s["box_pose_xyz_xyzw"], len(s["grasp_poses_xyz_xyzw"])
        print(f"\nRequest {s['request_id']}: '{s['object']['label']}' (object {s['object']['id']}, "
              f"score {s['object']['score']}, {s['object']['points']} points) in {s['frame']}")
        print(f"Box: centre ({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}) m, size "
              f"{' x '.join(f'{v:.3f}' for v in s['box_size_m'])} m")
        print(f"{n} grasp and {len(s['pregrasp_poses_xyz_xyzw'])} pre-grasp poses published "
              f"in {s['seconds']} s; saved to {s['saved_to']}", flush=True)
    except KeyboardInterrupt:
        code = 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 2
    try:
        if code == 0:
            if args.hold is None:
                print("Keeping the poses available; press Ctrl+C to stop.", flush=True)
            end = None if args.hold is None else time.monotonic() + args.hold
            while rclpy.ok() and (end is None or time.monotonic() < end):
                rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, Exception):
        pass  
        # Ctrl+C ends the wait; ROS may report it as either
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
