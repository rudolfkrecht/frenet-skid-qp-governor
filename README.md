# Real-Time Torque-Constrained Motion Control for Skid-Steer Robots Using a QP-Based Command Governor

Rudolf Krecht and Áron Ballagi


## Quick start

Python 3.10 or later. The QP core uses only the standard library. For analysis
and numerical tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/evaluate_bags.py
python tools/make_figures.py
```

`docs/index.html` works directly from disk and on GitHub Pages. It requires no
build step, remote fonts, CDNs, or analytics.

## QP formulation

The command is u=(v,r), measured state x=(v_meas,r_meas), and previous output
u_prev. The objective is

`w_track_v*(v-v_nom)^2 + w_track_r*(r-r_nom)^2`
`+ w_smooth_v*(v-v_prev)^2 + w_smooth_r*(r-r_prev)^2`.

With measured state held fixed, the one-step inverse model is affine in u:

```
tau_sum  = ((v-v_meas)/Ts + f_c*tanh(v_meas/v_eps)
            + a_v*v_meas + g_gamma) / b_v
tau_diff = ((r-r_meas)/Ts + a_r*r_meas) / b_r
tau_left  = (tau_sum-tau_diff)/2
tau_right = (tau_sum+tau_diff)/2
```

Linear inequalities enforce forward speed, yaw rate, slew, curvature, and
aggregate/differential/side effort limits. Optional nonreversing wheel-speed
constraints are supported. The forward-only curvature cone does not implement
reverse tracking or in-place rotation. A diagonal positive-definite Hessian
allows an exact 2-D active-set candidate enumeration. Infeasibility returns no
command rather than silently claiming a violating optimum is feasible.

The method is model-based command feasibility, not a guarantee of actual
motor torque, traction, or terrain-independent safety. No post-QP filtering is
applied, because that could invalidate the constraints.

## ROS 2 adapter

Requires ROS 2 with rclpy, geometry_msgs, nav_msgs and std_msgs. Run from the
repository root in a sourced ROS environment:

```bash
python -m governor.ros_node --ros-args --params-file config/reference_demo.yaml
```

Default output is `/governor/cmd_vel`. Check model calibration, units, topic
frames/signs and drivetrain limits before connecting this output to a robot.
Parameters are read once at startup. Command or odometry timeout, invalid input,
or an infeasible QP results in a zero **stop request**, accompanied by a JSON
status on `/governor/debug`. A zero request is not asserted to satisfy the
one-step effort or slew constraints; stopping and emergency protection require
their own validated hardware-level mechanisms.

## Data and evaluation

Each verified bag contains `/tyr/odom`, `/generated_path`, `/cmd_vel_mpc`, and
`/tyr/cmd_vel`. Odometry and reference are in `map`. CTE is the shortest unsigned
distance to the reference polyline, using all odometry samples, with no
alignment or time trimming. The analysis script verifies SQLite integrity and
frame consistency. Display traces are decimated; reported metrics are not.

| Run | RMS CTE (m) | Maximum CTE (m) | Odometry samples |
|---|---:|---:|---:|
| Path 1, rep2 | 0.0053055873 | 0.0083691376 | 5267 |
| Path 2, rep2 | 0.0160956702 | 0.0305717924 | 9347 |
| Path 3, rep2 | 0.0248169188 | 0.0389112179 | 6268 |

## Identification material

`tools/identification_original.py` reads motor controller (in this case ODrive)
controller_status and IMU (in this case VectorNav) messages, calibrates axis signs, and fits
longitudinal and yaw channels. It requires ROS 2 and the original custom
ODrive message definitions; these custom definitions are not bundled.

`config/identified_params_original.json` contains the parameters of the robot platform. By default, it records m=400 kg,
Rw=0.169 m, b=0.716 m, Iz=200 kg m² and 23 input runs. 



