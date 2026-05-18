# identification_2 Simulator

Generic simulator entrypoint (not robot-owned), fully implemented inside `identification_2/simulator`.

- `N` models via `n_model`
- `M` env per model via `cfg.nworld`
- standalone runtime (no `Robot` inheritance)
- tensor-first APIs (`send_action_tensor`, `get_observation_tensor`)

## Quick use

```python
from identification_2.simulator import HumanoidMJWarpConfig, HumanoidMJWarpModelPool

cfg = HumanoidMJWarpConfig(
    # mjcf_path is optional; default path is resolved from model dependency
    # by identification_2.simulator.mjcf_paths
    nworld=3,
    device="cpu",
    fixed_base=True,
)

pool = HumanoidMJWarpModelPool(cfg, n_model=8, use_multiprocessing=True, mp_start_method="spawn")
pool.connect()

# action shape: (N, M, 12)
# pool.send_action_tensor(action_deg)

# parameter APIs:
# pool.set_joint_gains_per_model(...)
# pool.set_joint_armature_per_model(...)
# pool.set_joint_viscous_friction_per_model(...)
# pool.set_joint_static_friction_per_model(...)
# pool.set_body_mass_per_model(...)
# pool.set_body_com_per_model(...)

pool.disconnect()
```
