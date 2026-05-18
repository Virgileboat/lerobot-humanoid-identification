from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from identification_2.simulator import JOINT_ORDER, HumanoidMJWarpConfig, HumanoidMJWarpModelPool


def _validate_joint(joint: str) -> str:
    key = str(joint)
    if key not in JOINT_ORDER:
        raise ValueError(f"Unknown joint '{joint}'. Expected one of: {list(JOINT_ORDER)}")
    return key


def _action_to_vec(action_deg: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(action_deg, Mapping):
        out = np.zeros((len(JOINT_ORDER),), dtype=np.float32)
        for k, v in action_deg.items():
            name = str(k)
            if name.endswith(".pos"):
                name = name[: -len(".pos")]
            if name not in JOINT_ORDER:
                raise ValueError(f"Unknown action key '{k}'. Expected joint key or '<joint>.pos'.")
            out[int(JOINT_ORDER.index(name))] = float(v)
        return out

    arr = np.asarray(action_deg, dtype=np.float32).reshape(-1)
    if arr.shape != (len(JOINT_ORDER),):
        raise ValueError(f"Expected action size {len(JOINT_ORDER)}, got {arr.shape}")
    return arr


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class SingleEnvSimHelper:
    """
    Manual debug helper around the identification_2 simulator stack.

    - n_model = 1
    - nworld = 1
    """

    def __init__(
        self,
        *,
        mjcf_path: str | Path | None = None,
        device: str = "cpu",
        fixed_base: bool = True,
        sim_dt: float = 0.005,
        physics_substeps_per_action: int = 1,
        use_multiprocessing: bool = False,
        mp_start_method: str = "spawn",
    ):
        cfg_kwargs: dict[str, Any] = {
            "nworld": 1,
            "device": str(device),
            "fixed_base": bool(fixed_base),
            "sim_dt": float(sim_dt),
            "physics_substeps_per_action": int(max(1, physics_substeps_per_action)),
        }
        if mjcf_path is not None:
            cfg_kwargs["mjcf_path"] = Path(mjcf_path)
        cfg = HumanoidMJWarpConfig(**cfg_kwargs)
        self.pool = HumanoidMJWarpModelPool(
            cfg,
            n_model=1,
            use_multiprocessing=bool(use_multiprocessing),
            mp_start_method=str(mp_start_method),
        )

    @property
    def joints(self) -> list[str]:
        return list(JOINT_ORDER)

    @property
    def is_connected(self) -> bool:
        return bool(self.pool.is_connected)

    def connect(self) -> "SingleEnvSimHelper":
        if not self.pool.is_connected:
            self.pool.connect()
        return self

    def disconnect(self) -> None:
        if self.pool.is_connected:
            self.pool.disconnect()

    def reset(self) -> None:
        self.pool.reset()

    def step(self, nstep: int = 1) -> None:
        self.pool.step(nstep=int(max(1, nstep)))

    def send_action(
        self,
        action_deg: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        step: bool = True,
    ) -> np.ndarray:
        act = _action_to_vec(action_deg)
        self.pool.send_action_tensor(act.reshape(1, -1), step=bool(step))
        return act

    def get_observation(self) -> dict[str, np.ndarray]:
        obs = self.pool.get_observation_tensor()
        out: dict[str, np.ndarray] = {}
        for k, v in obs.items():
            arr = _to_numpy(v)
            if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == 1:
                out[k] = arr[0, 0]
            elif arr.ndim >= 1 and arr.shape[0] == 1:
                out[k] = arr[0]
            else:
                out[k] = arr
        return out

    def zero_action(self) -> np.ndarray:
        return np.zeros((len(JOINT_ORDER),), dtype=np.float32)

    def set_single_joint_action(self, joint: str, value_deg: float, *, step: bool = True) -> np.ndarray:
        j = _validate_joint(joint)
        action = self.zero_action()
        action[int(JOINT_ORDER.index(j))] = float(value_deg)
        return self.send_action(action, step=step)

    def get_model_snapshot(self) -> dict[str, Any]:
        return self.pool.get_model_snapshot(0)

    def get_joint_runtime_params(self, joint: str) -> dict[str, float]:
        j = _validate_joint(joint)
        rows = self.pool.get_runtime_joint_params(joint=j, model_indices=[0])
        return dict(rows[0])

    def __enter__(self) -> "SingleEnvSimHelper":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.disconnect()


def make_single_env_sim_helper(
    *,
    mjcf_path: str | Path | None = None,
    device: str = "cpu",
    fixed_base: bool = True,
    sim_dt: float = 0.005,
    physics_substeps_per_action: int = 1,
    use_multiprocessing: bool = False,
    mp_start_method: str = "spawn",
) -> SingleEnvSimHelper:
    """
    Convenience constructor for IPython:
        sim = make_single_env_sim_helper().connect()
    """
    return SingleEnvSimHelper(
        mjcf_path=mjcf_path,
        device=device,
        fixed_base=fixed_base,
        sim_dt=sim_dt,
        physics_substeps_per_action=physics_substeps_per_action,
        use_multiprocessing=use_multiprocessing,
        mp_start_method=mp_start_method,
    )

