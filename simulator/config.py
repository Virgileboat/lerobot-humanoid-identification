from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from simulator.robot_spec import JOINT_ORDER as HUMANOID_JOINT_ORDER
from simulator.mjcf_paths import get_default_mjcf_path

JOINT_ORDER = tuple(str(j) for j in HUMANOID_JOINT_ORDER)


@dataclass(kw_only=True)
class HumanoidMJWarpConfig:
    # Optional identifier used to derive per-model IDs in the pool.
    id: str | None = None

    # Default to the humanoid scene in identification_2 models.
    mjcf_path: Path = field(default_factory=lambda: get_default_mjcf_path(fixed_base=False))

    # Batched simulation.
    nworld: int = 128
    sim_dt: float = 0.005
    physics_substeps_per_action: int = 1
    fixed_base: bool = False

    # Device used by torch/warp arrays.
    device: str = "cuda"

    # Optional reset to initial state on connect.
    reset_on_connect: bool = True
