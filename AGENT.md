# AGENT.md

## Remote Reference

- Remote repository: <https://github.com/Virgileboat/lerobot-humanoid-identification>
- Default branch: <https://github.com/Virgileboat/lerobot-humanoid-identification/tree/main>
- Current tracked upstream branch: `origin/main`

## Full Project Context

This repository is the identification block in the LeRobot humanoid stack:

1. `lerobot-humanoid-design`: co-design assumptions and feasibility
2. `lerobot-humanoid-hardware`: real build and commissioning procedures
3. `lerobot-humanoid-model`: shared MJCF/URDF assets
4. `lerobot-humanoid-runtime`: real/sim control and data acquisition
5. `lerobot-humanoid-identification`: offline replay + parameter identification (this repo)

Identification outputs are used to tune simulator realism for runtime and training loops.

## Mission Of This Repo

Run joint-wise simulator identification with MJWarp + CMA-ES:

- load recorded datasets
- replay trajectories in batched simulation
- optimize model/dynamics parameters per joint
- export reproducible run artifacts

## Core Pipeline (Do Not Break)

1. `cmaes/run_cmaes_pool_all_joints.py`
2. `cmaes/parallel_jointwise_cmaes.py`
3. `cmaes/pool_joint_cost.py`
4. `simulator/runtime.py`
5. result export to `results/.../pool_jointwise_<timestamp>/`

## Critical Contracts

- CLI stays root-runnable (`uv run python -m cmaes...`).
- Dataset layout compatibility remains (`grouped_by_joint`, `per_experiment`, `auto`).
- Fixed-base resolution behavior stays coherent with `simulator/mjcf_paths.py`.
- Parameter write APIs in `simulator/runtime.py` remain synchronized with cost code.
- `JOINT_ORDER` assumptions stay explicit and backward-compatible.

## Dependencies And Interfaces

- Model dependency is the in-repo submodule path `lerobot-humanoid-models/`.
- Upstream data often comes from runtime acquisition workflows.
- Downstream consumers include runtime/training experiments using identified params.

## Validation Before Merge

Run at least one smoke identification command and verify:

- process completes
- `summary.yaml` exists
- per-joint result files are generated

Preferred smoke shape: CPU, small `popsize`, `maxiter=1`, single joint.
