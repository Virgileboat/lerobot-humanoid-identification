# AGENT.md - Working Contract (Virgile style)

## Identity and Scope

This repo is the identification engine for LeRobot Humanoid:

- replay many datasets/configurations in parallel
- optimize simulator parameters with joint-wise CMA-ES

Out of scope here: real robot acquisition/control stacks.  
Those live in other repos, but this repo must stay compatible with what they produce.

## Full Project Context

This repo is one block in the LeRobot humanoid stack:

1. **Model geometry & MJCF**: `lerobot-humanoid-models` (submodule, version pinned)
2. **Robot runtime/control repos**: e.g. `lerobot_humanoid_runtime`, `lerobot_humanoide_hardware`, `lerobot_can`
3. **Policy/training repos**: e.g. `lerobot`, `lerobot_huamnoide_training`
4. **This repo**: offline replay + sim-ID + reference outputs for regression
5. **Downstream loop**: tuned sim params fed back to runtime/training experiments

Rule: any change here must preserve compatibility with 1/2/3.

## Non-Negotiables

- Keep the core loop stable:
  - dataset load -> replay pool -> CMA-ES -> per-joint outputs
- Keep release commands root-runnable (`uv run python -m cmaes...`).
- Keep fixed-base behavior deterministic for release runs.
- Keep minimal, clean release artifacts (no random debug leftovers).
- Keep compatibility with existing dataset layout and `JOINT_ORDER`.
- Never silently break command lines used in README and release notes.

## Critical Path

1. `cmaes/run_cmaes_pool_all_joints.py:main()`
2. `cmaes/parallel_jointwise_cmaes.py:_run_one_joint()`
3. `cmaes/pool_joint_cost.py:cost_for_joint()`
4. `cmaes/pool_joint_cost.py:_apply_joint_parameters()`  <- most critical write path
5. `simulator/runtime.py:HumanoidMJWarpModelPool.replay_datasets_actions()`
6. LSQ aggregation back to CMA-ES

Any bug here invalidates final identified parameters.

## Fixed-Base Contract

- User asks `--fixed-base`.
- Runtime resolves to a fixed-base scene (`*_fixed_base.xml`), even if config stores `scene.xml`.
- Do not assume saved config `mjcf_path` equals runtime-loaded file path.

## Parameter Write APIs (must stay coherent)

- `set_joint_armature_per_model`
- `set_joint_viscous_friction_per_model`
- `set_joint_static_friction_per_model`
- `set_joint_torque_limit_per_model`
- `set_joint_action_delay_steps_per_model`
- `set_joint_gains_per_model`
- `set_body_mass_per_model`
- `set_body_com_per_model`

## Release Baseline Assets

- Dataset bundle kept in repo:
  - `models/lerobot_humanoid/datasets/baseline_controller_v1`
- Reference result kept in repo:
  - `results/baseline_controller_v1/pool_jointwise_20260403_132618`
- Model dependency:
  - `lerobot-humanoid-models` submodule

## Required Smoke Test Before Merge/Release

```bash
uv run python -m cmaes.run_cmaes_pool_all_joints \
  --datasets-root models/lerobot_humanoid/datasets/baseline_controller_v1 \
  --experiments experiment_2s_step_inv \
  --dataset-layout auto \
  --fixed-base \
  --device cpu \
  --joint-order right_knee \
  --popsize 2 --maxiter 1 --sigma0 0.3 \
  --out-dir results/smoke_cpu_release
```

Expected:

- command completes
- `summary.yaml` exists
- warnings may exist (`nefc overflow`), but no crash

## Engineering Style for This Repo

- Keep changes focused and reviewable.
- Prefer explicit paths and explicit contracts over hidden behavior.
- If uncertain, run smoke test and report exact files/outputs.
