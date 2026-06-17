# Implementation ground truth (pinned from on-disk Isaac Lab + trossen + the USD)

Source: `~/isaac-rl/IsaacLab/source` (IL), `~/isaac-rl/trossen_ai_isaac` (TR), rsl_rl in container `/opt/venv` (RSL), `stationary_ai.usd` joint dump (USD). Pinned 2026-06-16. Use these EXACT values; items marked UNVERIFIED need a check.

## Stationary AI joints (USD ground truth)
- LEFT arm: `follower_left_joint_[0-5]` (revolute) · LEFT gripper: `follower_left_left_carriage_joint` + `follower_left_right_carriage_joint` (prismatic)
- RIGHT arm: `follower_right_joint_[0-5]` · RIGHT gripper: `follower_right_.*_carriage_joint`
- EE bodies: `follower_left_ee_gripper_link`, `follower_right_ee_gripper_link` · arm bases: `follower_left_base_link`, `follower_right_base_link`
- Camera mounts in USD: `cam_high_link`, `cam_low_link`, `follower_{left,right}_camera_link` (4× D405)
- 16 articulated DOF (12 arm + 4 carriage). v1 ACTIVE arm = LEFT; right parked.
- Carriage convention (WXAI): actuate only the LEFT carriage; right carriage is a USD mimic. UNVERIFIED for stationary_ai.usd → smoke test confirms it loads.

## Key corrections to the plan (CRITICAL)
1. PPO `obs_groups` key is **`"actor"`/`"critic"`**, not `"policy"`. `"policy"` is an env obs-group NAME (a value). → `{"actor": ["policy"], "critic": ["policy","privileged"]}` (RSL/algorithms/ppo.py).
2. Distillation `obs_groups` keys are **`"student"`/`"teacher"`** → `{"student": ["policy","images"], "teacher": ["policy","privileged"]}` (RSL/algorithms/distillation.py).
3. Termination param is **`minimum_height`** (root_height_below_minimum); REWARD funcs use `minimal_height`. Do not swap.
4. `object_position_in_robot_root_frame` is **lift-local** mdp (IL .../lift/mdp/observations.py), re-exported via the lift `mdp` namespace.
5. Image obs group must set `concatenate_terms=False`.
6. `RslRlPpoActorCriticCfg` is DEPRECATED (rsl-rl≥4) → use `actor=`/`critic=` `RslRlMLPModelCfg` + nested `GaussianDistributionCfg(init_std=, std_type=)`.
7. Image is **NHWC** `(N,H,W,C)`; permute `(0,3,1,2)` before the CNN. Whether CNNModel permutes internally: UNVERIFIED.

## Reward weights (IL .../lift/lift_env_cfg.py, EXACT)
- reaching_object `object_ee_distance` w=1.0 {std:0.1}
- lifting_object `object_is_lifted` w=15.0 {minimal_height:0.04}
- object_goal_tracking `object_goal_distance` w=16.0 {std:0.3, minimal_height:0.04, command_name:"object_pose"}
- object_goal_tracking_fine_grained `object_goal_distance` w=5.0 {std:0.05, minimal_height:0.04, command_name:"object_pose"}
- action_rate `action_rate_l2` w=-1e-4 · joint_vel `joint_vel_l2` w=-1e-4 {asset_cfg robot}
- curriculum: action_rate/joint_vel → -1e-1 at num_steps=10000

## Terminations / events / command / actions / timing
- term: `time_out`(time_out=True); `object_dropping` `root_height_below_minimum` {minimum_height:-0.05, asset_cfg object}
- events: `reset_scene_to_default`(reset); `reset_root_state_uniform`(reset) {pose_range x:(-0.1,0.1) y:(-0.25,0.25) z:(0,0), velocity_range {}, asset_cfg object body_names "Object"}
- command `object_pose` UniformPoseCommandCfg resampling_time_range (5,5); WXAI ranges pos_x(0.2,0.4) pos_y(-0.15,0.15) pos_z(0.1,0.3); body_name "follower_left_ee_gripper_link" (UNVERIFIED reach → tune)
- actions: arm `JointPositionActionCfg` joint_names ["follower_left_joint_[0-5]"] scale=0.5 use_default_offset=True; gripper `BinaryJointPositionActionCfg` joint_names ["follower_left_left_carriage_joint"] open=0.044 close=0.0 (0.044 is WXAI stroke, UNVERIFIED)
- sim: decimation=2, episode_length_s=5.0, sim.dt=0.01 (inherited from LiftEnvCfg)
- object DexCube: `{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd` scale (0.8,0.8,0.8); WXAI init pos [0.3,0,0.055] (UNVERIFIED reach → tune)
- ee_frame sensor MUST be named `ee_frame`; source follower_left_base_link, target follower_left_ee_gripper_link

## TiledCamera / mdp.image (vision)
- TiledCameraCfg(prim_path "{ENV_REGEX_NS}/Robot/cam_high_link", offset OffsetCfg(pos,rot(w,x,y,z),convention="ros"), data_types ["distance_to_camera"], spawn PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.01,1e6)), width/height, update_period=0.0). Camera offset UNVERIFIED.
- mdp.image(sensor_cfg, data_type, normalize=True): depth → inf→0 (NO min-max). Output NHWC; permute to NCHW.

## rsl_rl cfg classes (IL .../isaaclab_rl/rsl_rl/rl_cfg.py, distillation_cfg.py)
- PPO: RslRlOnPolicyRunnerCfg(class_name "OnPolicyRunner"; actor/critic RslRlMLPModelCfg; algorithm RslRlPpoAlgorithmCfg); obs_groups {"actor":["policy"],"critic":["policy","privileged"]}
- RslRlMLPModelCfg(class_name "MLPModel"; hidden_dims; activation; obs_normalization=False; distribution_cfg GaussianDistributionCfg(init_std, std_type "scalar"))
- PPO algo (WXAI hyperparams): value_loss_coef=1.0, clip_param=0.2, entropy_coef=0.006, num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-4, schedule="adaptive", gamma=0.98, lam=0.95, desired_kl=0.01, max_grad_norm=1.0; num_steps_per_env=24, max_iterations=1500
- Distill: RslRlDistillationRunnerCfg(class_name "DistillationRunner"; student/teacher RslRlMLPModelCfg; algorithm RslRlDistillationAlgorithmCfg(class_name "Distillation"; num_learning_epochs; learning_rate; gradient_length; max_grad_norm; optimizer; loss_type "mse"))
- RslRlCNNModelCfg(extends MLP; class_name "CNNModel"; cnn_cfg CNNCfg(output_channels, kernel_size, stride=1, padding, norm, activation, max_pool=False, global_pool, flatten=True))

## Still UNVERIFIED (resolve during impl)
asset path/depth · carriage mimic · gripper open stroke · camera offset · cube pos + command ranges (left-arm reach) · table reposition · CNN permute internal? · CNN channels/kernels for 80×80.
