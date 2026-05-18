# Identification_2 Class Diagram

This diagram is a proper class diagram for the core replay + identification path.
Rendered SVG: `docs/class_diagram.svg`.

```mermaid
classDiagram
  class run_cmaes_pool_all_joints {
    +main()
    +_validate_joint_action_excitation()
    +_save_joint_artifacts(...)
  }

  class JointwiseExperimentData {
    +experiment_names: list[str]
    +dataset_layout: str
    +datasets_by_joint: dict[str, list[JointMotionDataset]]
    +base_kp: np.ndarray
    +base_kv: np.ndarray
  }

  class JointMotionDataset {
    +actions_deg: np.ndarray
    +observed_pos_deg: np.ndarray
    +timestamps_s: np.ndarray
  }

  class JointwiseCMAESConfig {
    +popsize: int
    +maxiter: int
    +sigma0: float
    +joint_order: tuple[str]
  }

  class ParallelJointwiseCMAES {
    +run() JointwiseCMAESResult
    +_run_one_joint(...) dict
  }

  class MJWarpPopulationJointCost {
    +cost_for_joint(joint, candidate_params) np.ndarray
    +apply_candidate_batch_for_joint(joint, candidate_params) np.ndarray
    +replay_joint_datasets(joint) dict
    +_apply_joint_parameters(active_joint, candidate_mat)
    +_lsq_per_model_for_joint(...) np.ndarray
  }

  class HumanoidMJWarpModelPool {
    +connect()
    +prepare_replay_datasets(datasets, cache_key) str
    +replay_datasets_actions(...) dict
    +set_joint_armature_per_model(...)
    +set_joint_viscous_friction_per_model(...)
    +set_joint_static_friction_per_model(...)
    +set_joint_torque_limit_per_model(...)
    +set_joint_action_delay_steps_per_model(...)
    +set_joint_gains_per_model(...)
    +set_body_mass_per_model(...)
    +set_body_com_per_model(...)
  }

  class HumanoidMJWarp {
    +connect()
    +reset()
    +send_action_tensor(action_deg)
    +get_observation_tensor() dict
  }

  class mjcf_paths {
    +get_model_mjcf_dir()
    +get_default_mjcf_path(fixed_base) Path
    +ensure_fixed_base_scene(scene_xml) Path
  }

  run_cmaes_pool_all_joints --> JointwiseExperimentData : load_jointwise_experiment_data()
  JointwiseExperimentData --> JointMotionDataset : contains
  run_cmaes_pool_all_joints --> HumanoidMJWarpModelPool : build pool
  run_cmaes_pool_all_joints --> MJWarpPopulationJointCost : build evaluator
  run_cmaes_pool_all_joints --> ParallelJointwiseCMAES : run optimizer
  ParallelJointwiseCMAES --> JointwiseCMAESConfig : uses
  ParallelJointwiseCMAES --> MJWarpPopulationJointCost : evaluate population
  MJWarpPopulationJointCost --> HumanoidMJWarpModelPool : apply params + replay
  HumanoidMJWarpModelPool --> HumanoidMJWarp : owns N models
  HumanoidMJWarp --> mjcf_paths : resolves scene path
```

## Call Contracts

- `JointMotionDataset.actions_deg`: shape `(T, 12)`, order must match `JOINT_ORDER`.
- Canonical `JOINT_ORDER` and baseline gains source: `simulator/robot_spec.py`.
- Population matrix contract in CMAES: `candidate_params.shape == (popsize, n_param)`.
- Replay contract: `len(datasets) == nworld`.
- Cost return contract: LSQ vector shape `(popsize,)`, one cost per model candidate.
