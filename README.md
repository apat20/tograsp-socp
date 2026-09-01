# tograsp-socp

Python implementation for computing a task-dependent grasp metric as a second-order cone program (SOCP).

A task is formalized in terms of the **constant screw motion** to be imparted to the object *after* grasping. Given a (partial) point cloud of an object and a task screw, this code computes the metric for antipodal contacts sampled on the object's bounding box, extracts the resulting grasping region, and computes candidate 6-DOF end-effector poses.

Please note that this repository is under active development.

## Installation

Requires Python 3.10. With conda:

```bash
conda env create -f environment.yml
conda activate tograsp_socp
```

Or with a plain virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

Each run takes roughly 10 seconds on the bundled point cloud and opens two matplotlib windows: the object bounding box with the computed end-effector frames, and the same box with the grasp centers marked. Add `--no-plot` on a headless machine.

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--filename` | `nontextured.ply` | Path to the object point cloud (`.ply`) |
| `--trial` | `1` | Label for this run; determines the output folder name |
| `--log-dir` | `logs` | Parent directory for the per-trial output folders |
| `--no-plot` | off | Skip the matplotlib windows |

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

## What the parameters mean

The physical parameters currently live in the `__main__` block and in `build_and_solve_gfop` of each script. The ones most likely to need changing for a different object or gripper:

| Parameter | Where | Default | Meaning |
| --- | --- | --- | --- |
| `grasp.gripper_width_tolerance` | `__main__` | 0.08 m | Max gripper opening; decides which pair of bounding box faces the contacts are sampled on |
| `grasp.gripper_height_tolerance` | `__main__` | 0.041 m | Used to reject approach directions |
| `grasp.grasp_metric_threshold` | `__main__` | 0.7 | Fraction of the max metric above which a grid cell joins the grasping region |
| `gfop_object.F` | `build_and_solve_gfop` | 30 N | Max normal force per finger |
| `gfop_object.F_external` | `build_and_solve_gfop` | 10 N along -z | External wrench on the object (its weight) |
| `gfop_object.mu1`, `mu2` | `build_and_solve_gfop` | 0.4 | Friction coefficients at the object-robot contacts |
| `gfop_object.mu` | `build_and_solve_gfop` | 0.6 | Friction coefficient at the object-environment contact |

The end-effector offsets in `get_end_effector_poses` are specific to a Franka Emika Panda with the stock fingers.

## Repository layout

```
main_pickup.py            Entry point: pure translation task screw
main_gcsm.py              Entry point: general constant screw motion
socp_module/              The SOCP itself (grasp map, friction cones, solve)
point_cloud_module/       Point cloud processing, bounding boxes, grasping region,
                          end-effector pose computation
func/                     Quaternion / dual quaternion / ScLERP utilities, plotting helpers
matlab_helper_funcs/      MATLAB plotting helpers
visualize_results.m       MATLAB visualization of a logged trial
nontextured.ply           Example object point cloud
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