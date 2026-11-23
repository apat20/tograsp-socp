# tograsp-socp

Python implementation for computing a task-dependent grasp metric as a second-order cone program (SOCP).

A task is formalized in terms of the **constant screw motion** to be imparted to the object *after* grasping. Given a (partial) point cloud of an object and a task screw, this code computes the metric for antipodal contacts sampled on the object's bounding box, extracts the resulting grasping region, and computes candidate 6-DOF end-effector poses.

This branch has the standalone code. To run it on a robot with ROS 2 and Docker (for example on an NVIDIA Jetson Orin, next to a perception container), see the [`ros2-docker`](https://github.com/apat20/tograsp-socp/tree/ros2-docker) branch.

Please note that this repository is under active development.

## Installation

With conda (Python 3.13, all packages from conda-forge):

```bash
conda env create -f environment.yml
conda activate tograsp_socp
```

The dependencies are `numpy`, `scipy`, `cvxpy` (with its default solver), `open3d`, and `matplotlib`.

## Quick start

Two entry points, one per category of constant screw motion:

| Script | Task screw | Metric computed |
| --- | --- | --- |
| `main_pickup.py` | Pure translation (pitch = infinity) | Max force along the screw axis |
| `main_gcsm.py` | General constant screw motion | Max moment about the screw axis |

Run either from the repository root:

```bash
python main_pickup.py --filename nontextured.ply --trial 1
python main_gcsm.py   --filename nontextured.ply --trial 1
```

Each run takes roughly 10 seconds on the bundled point cloud and opens two matplotlib windows: the object bounding box with the computed end-effector frames, and the same box with the grasp centers marked. On a machine without a display, start the command with `MPLBACKEND=Agg` to skip the windows.

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--filename` | `nontextured.ply` | Path to the object point cloud (`.ply`) |
| `--trial` | `1` | Label for this run; determines the output folder name |
| `--log-dir` | `logs` | Parent directory for the per-trial output folders |

### Output

Results are written to `<log-dir>/pickup_trial_<trial>/` or `<log-dir>/gcsm_trial_<trial>/`. The directory is created automatically. Contents:

- `initial_cloud_object_frame.ply`, `projected_object_frame.ply` — the input cloud and its projection onto a bounding box face, both in the object frame
- `unit_vector.csv`, `point.csv` — the task screw axis
- `transformed_vertices_object_frame.csv` — bounding box vertices in the object frame
- `x_data.csv`, `test_datapoints.csv` — sampled antipodal contacts and screw parameters, one row per contact pair
- `test_predicted.csv`, `test_metric_values.csv` — the computed metric, normalized to [0, 1]
- `metric_grid_computed.csv`, `metric_grid_occupied_computed.csv` — metric values mapped onto the projection grid
- `X/Y/Z_grid_points_*.csv`, `q_*_array_computed.csv` — grid geometry used by the MATLAB visualization

Outputs are ignored by git. `visualize_results.m` reads a trial folder for the MATLAB-side plots.

## Parameters

The physical parameters are in `config.py`. The ones most likely to need changing for a different object or gripper:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `GRIPPER_WIDTH_TOLERANCE` | 0.08 m | Max gripper opening; decides which pair of bounding box faces the contacts are sampled on |
| `GRIPPER_HEIGHT_TOLERANCE` | 0.041 m | Used to reject approach directions |
| `G_DELTA`, `G_DELTA_INTER` | 0.1434 m, 0.18 m | Distance from the grasp center back to the flange, for the grasp and pre-grasp poses |
| `F` | 30 N | Max normal force per finger |
| `F_EXTERNAL` | 10 N along -z | External wrench on the object (its weight) |
| `MU1`, `MU2` | 0.4 | Friction coefficients at the object-robot contacts |

The default `G_DELTA` and `G_DELTA_INTER` are for a Franka Emika Panda with the stock fingers. The threshold that decides which grid cells join the grasping region (0.7 of the maximum metric) is set in `run_pickup()` and `run_gcsm()`.

## Repository Structure

```
main_pickup.py            Entry point: pure translation task screw
main_gcsm.py              Entry point: general constant screw motion
pickup_runner.py          run_pickup(): the computation behind main_pickup.py, without plotting
gcsm_runner.py            run_gcsm(): the computation behind main_gcsm.py, without plotting
config.py                 SOCP and gripper parameters
socp_module/              The SOCP itself (grasp map, friction cones, solve)
point_cloud_module/       Point cloud processing, bounding boxes, grasping region,
                          end-effector pose computation
func/                     Quaternion / dual quaternion / ScLERP utilities, plotting helpers
matlab_helper_funcs/      MATLAB plotting helpers
visualize_results.m       MATLAB visualization of a logged trial
nontextured.ply           Example object point cloud
environment.yml           Conda environment
```

## Citation

Initial IROS 2021 paper:

```
@inproceedings{fakhari2021computing,
  title={Computing a task-dependent grasp metric using second-order cone programs},
  author={Fakhari, Amin and Patankar, Aditya and Xie, Jiayin and Chakraborty, Nilanjan},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4009--4016},
  year={2021},
  organization={IEEE}
}
```

Related papers:

```
@inproceedings{patankar2023task,
  title={Task-Oriented Grasping with Point Cloud Representation of Objects},
  author={Patankar, Aditya and Phi, Khiem and Mahalingam, Dasharadhan and Chakraborty, Nilanjan and Ramakrishnan, IV},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={6853--6860},
  year={2023},
  organization={IEEE}
}
```

```
@inproceedings{fakhari2021motion,
  title={Motion and force planning for manipulating heavy objects by pivoting},
  author={Fakhari, Amin and Patankar, Aditya and Chakraborty, Nilanjan},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={9393--9400},
  year={2021},
  organization={IEEE}
}
```

A journal version of this paper is under preparation.

## License

Apache-2.0. See [LICENSE](LICENSE).