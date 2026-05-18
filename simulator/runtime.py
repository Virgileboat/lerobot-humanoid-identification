#!/usr/bin/env python

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from simulator.config import JOINT_ORDER as CONFIG_JOINT_ORDER, HumanoidMJWarpConfig

JOINT_ORDER = list(CONFIG_JOINT_ORDER)
JOINT_INDEX = {j: i for i, j in enumerate(JOINT_ORDER)}

_MJ_JOINT_NAME_BY_KEY = {
    "left_hipz": "hipz_left",
    "left_hipx": "hipx_left",
    "left_hipy": "hipy_left",
    "left_knee": "knee_left",
    "left_ankle_pitch": "ankley_left",
    "left_ankle_roll": "anklex_left",
    "right_hipz": "hipz_right",
    "right_hipx": "hipx_right",
    "right_hipy": "hipy_right",
    "right_knee": "knee_right",
    "right_ankle_pitch": "ankley_right",
    "right_ankle_roll": "anklex_right",
}

_MJ_ACTUATOR_NAME_BY_KEY = {
    "left_hipz": "m_hipz_left",
    "left_hipx": "m_hipx_left",
    "left_hipy": "m_hipy_left",
    "left_knee": "m_knee_left",
    "left_ankle_pitch": "m_ankley_left",
    "left_ankle_roll": "m_anklex_left",
    "right_hipz": "m_hipz_right",
    "right_hipx": "m_hipx_right",
    "right_hipy": "m_hipy_right",
    "right_knee": "m_knee_right",
    "right_ankle_pitch": "m_ankley_right",
    "right_ankle_roll": "m_anklex_right",
}


def _actuator_torque_limit_or_default(mjm: Any, aid: int, *, default: float = 1.0) -> float:
    """Return a positive finite torque limit even when MJCF has unset/zero force range."""
    lo = float(mjm.actuator_forcerange[aid, 0])
    hi = float(mjm.actuator_forcerange[aid, 1])
    lim = max(abs(lo), abs(hi))
    if np.isfinite(lim) and lim > 0.0:
        return float(lim)
    return float(default)


def _normalize_device(device: str) -> str:
    d = str(device).strip().lower()
    if d == "gpu":
        return "cuda"
    return d


def _available_cpu_ids() -> list[int]:
    try:
        return sorted(int(c) for c in os.sched_getaffinity(0))
    except Exception:
        n = int(os.cpu_count() or 1)
        return list(range(n))


def _physical_core_cpu_ids(available_cpu_ids: Iterable[int]) -> list[int]:
    """
    Return one logical CPU id per physical core (Linux sysfs topology).
    Falls back to the provided list if topology is unavailable.
    """
    selected_by_core: dict[tuple[int, int], int] = {}
    for cpu in sorted(int(c) for c in available_cpu_ids):
        core_path = f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
        pkg_path = f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
        try:
            with open(core_path, "r", encoding="utf-8") as f:
                core_id = int(f.read().strip())
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg_id = int(f.read().strip())
            key = (pkg_id, core_id)
        except Exception:
            key = (-1, cpu)
        prev = selected_by_core.get(key)
        if prev is None or cpu < prev:
            selected_by_core[key] = cpu

    out = sorted(int(v) for v in selected_by_core.values())
    return out if out else sorted(int(c) for c in available_cpu_ids)


def _pick_worker_cpu_ids(
    *,
    n_worker: int,
    pin_workers_to_cores: bool,
    prefer_physical_cores: bool,
) -> list[int | None]:
    if not bool(pin_workers_to_cores):
        return [None] * int(max(1, n_worker))
    available = _available_cpu_ids()
    if len(available) == 0:
        return [None] * int(max(1, n_worker))
    candidate = (
        _physical_core_cpu_ids(available_cpu_ids=available) if bool(prefer_physical_cores) else list(available)
    )
    if len(candidate) == 0:
        return [None] * int(max(1, n_worker))
    return [int(candidate[i % len(candidate)]) for i in range(int(max(1, n_worker)))]


def _resolve_mjcf_path(mjcf_path: Path, fixed_base: bool) -> Path:
    base = Path(mjcf_path)
    if not base.is_absolute():
        base = Path.cwd() / base
    base = base.resolve()
    if not fixed_base:
        if not base.exists():
            raise FileNotFoundError(f"Expected MJCF file not found: {base}")
        return base

    # First, try conventional suffix-based fixed-base scene.
    if base.name.endswith("_fixed_base.xml"):
        p = base
    else:
        p = base.with_name(f"{base.stem}_fixed_base{base.suffix}")
    if p.exists():
        return p

    # Fallback for external model dependency that only provides scene.xml + robot.xml.
    try:
        from simulator.mjcf_paths import ensure_fixed_base_scene

        if base.exists():
            p_gen = ensure_fixed_base_scene(base)
            if p_gen.exists():
                return p_gen
    except Exception:
        pass

    raise FileNotFoundError(f"Expected MJCF file not found: {p}")


class HumanoidMJWarp:
    """
    Minimal MJWarp humanoid wrapper:
    - one shared MuJoCo model
    - nworld parallel simulation data
    - actuator-position control only (ctrl = desired joint position in rad)
    """

    config_class = HumanoidMJWarpConfig
    name = "humanoid_mjwarp"

    def __init__(self, config: HumanoidMJWarpConfig):
        self.config = config
        self._connected = False

        try:
            import mujoco
            try:
                from mujoco import mjwarp as mjw
            except Exception:
                import mujoco_warp as mjw
            import torch
            import warp as wp
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HumanoidMJWarp requires mujoco + (mujoco.mjwarp or mujoco_warp) + warp + torch"
            ) from e

        self._mujoco = mujoco
        self._mjw = mjw
        self._torch = torch
        self._wp = wp

        self._device = _normalize_device(config.device)
        try:
            wp.set_device(self._device)
        except Exception:
            pass

        self._mjcf_path = _resolve_mjcf_path(Path(config.mjcf_path), bool(config.fixed_base))
        self._mjm = mujoco.MjModel.from_xml_path(str(self._mjcf_path))
        self._mjm.opt.timestep = float(config.sim_dt)

        self._nworld = int(max(1, config.nworld))
        self._substeps = int(max(1, config.physics_substeps_per_action))

        self._joint_qpos_idx: list[int] = []
        self._joint_qvel_idx: list[int] = []
        self._joint_mj_idx: list[int] = []
        self._actuator_idx: list[int] = []
        self._body_name_to_idx: dict[str, int] = {}
        self._build_indices()

        self._m = self._mjw.put_model(self._mjm)
        self._d = self._mjw.make_data(self._mjm, nworld=self._nworld)
        self._mjw.reset_data(self._m, self._d)

        self._ctrl = torch.zeros((self._nworld, int(self._mjm.nu)), device=self._device, dtype=torch.float32)
        self._actuator_idx_t = torch.as_tensor(self._actuator_idx, device=self._device, dtype=torch.long)
        self._joint_qpos_idx_t = torch.as_tensor(self._joint_qpos_idx, device=self._device, dtype=torch.long)
        self._joint_qvel_idx_t = torch.as_tensor(self._joint_qvel_idx, device=self._device, dtype=torch.long)
        self._sim_dt_s = float(self._mjm.opt.timestep)
        self._sim_time_s = 0.0
        self._replay_cache: dict[str, "_PreparedReplayBatch"] = {}
        self._action_delay_steps = np.zeros((len(JOINT_ORDER),), dtype=np.int64)

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Simulator is not connected. Call connect() first.")

    def _require_disconnected(self) -> None:
        if self._connected:
            raise RuntimeError("Simulator is already connected. Call disconnect() first.")

    def _build_indices(self) -> None:
        mj = self._mujoco
        for j in JOINT_ORDER:
            joint_name = _MJ_JOINT_NAME_BY_KEY[j]
            jid = int(mj.mj_name2id(self._mjm, mj.mjtObj.mjOBJ_JOINT, joint_name))
            if jid < 0:
                raise RuntimeError(f"Missing joint in MJCF: {joint_name}")
            self._joint_mj_idx.append(jid)
            self._joint_qpos_idx.append(int(self._mjm.jnt_qposadr[jid]))
            self._joint_qvel_idx.append(int(self._mjm.jnt_dofadr[jid]))

            actuator_name = _MJ_ACTUATOR_NAME_BY_KEY[j]
            aid = int(mj.mj_name2id(self._mjm, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name))
            if aid < 0:
                raise RuntimeError(f"Missing actuator in MJCF: {actuator_name}")
            self._actuator_idx.append(aid)

        for bid in range(int(self._mjm.nbody)):
            bname = mj.mj_id2name(self._mjm, mj.mjtObj.mjOBJ_BODY, bid)
            if bname:
                self._body_name_to_idx[bname] = bid

    def _sync_device_model(self) -> None:
        self._m = self._mjw.put_model(self._mjm)

    def _set_position_targets_deg(self, action_deg: Any) -> Any:
        torch = self._torch
        action = torch.as_tensor(action_deg, dtype=torch.float32, device=self._device)
        if action.ndim == 1:
            if int(action.shape[0]) != len(JOINT_ORDER):
                raise ValueError(f"Expected 12 joints, got {tuple(action.shape)}")
            action = action.unsqueeze(0).repeat(self._nworld, 1)
        if tuple(action.shape) != (self._nworld, len(JOINT_ORDER)):
            raise ValueError(f"Expected action shape {(self._nworld, len(JOINT_ORDER))}, got {tuple(action.shape)}")

        action_rad = action * (np.pi / 180.0)
        self._ctrl[:, self._actuator_idx_t] = action_rad
        return action

    @property
    def num_worlds(self) -> int:
        return self._nworld

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._require_disconnected()
        if self.config.reset_on_connect:
            self._reset_sim_state()
        else:
            self._sim_time_s = 0.0
        self._connected = True

    def _reset_sim_state(self) -> None:
        self._mjw.reset_data(self._m, self._d)
        self._ctrl.zero_()
        self._sim_time_s = 0.0

    def reset(self) -> None:
        self._require_connected()
        self._reset_sim_state()

    def step(self, nstep: int = 1) -> None:
        self._require_connected()
        n = int(max(1, nstep))
        self._d.ctrl = self._wp.from_torch(self._ctrl, dtype=self._wp.float32)
        for _ in range(n):
            self._mjw.step(self._m, self._d)
        self._sim_time_s += float(self._sim_dt_s) * float(n)

    def get_sim_time_s(self) -> float:
        self._require_connected()
        return float(self._sim_time_s)

    def send_action_tensor(self, action_deg: Any, step: bool = True) -> Any:
        self._require_connected()
        action = self._set_position_targets_deg(action_deg)
        if step:
            self.step(self._substeps)
        return action

    def get_observation_tensor(self) -> dict[str, Any]:
        self._require_connected()
        torch = self._torch
        qpos = self._wp.to_torch(self._d.qpos)
        qvel = self._wp.to_torch(self._d.qvel)

        pos_deg = torch.stack([qpos[:, idx] for idx in self._joint_qpos_idx], dim=1) * (180.0 / np.pi)
        vel_deg = torch.stack([qvel[:, idx] for idx in self._joint_qvel_idx], dim=1) * (180.0 / np.pi)
        out = {
            "joint_pos_deg": pos_deg,
            "joint_vel_deg_s": vel_deg,
        }

        if int(self._mjm.nq) >= 7:
            q = qpos[:, 3:7]
            out["base_orientation_xyzw"] = torch.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], dim=1)
        else:
            out["base_orientation_xyzw"] = torch.zeros((self._nworld, 4), dtype=torch.float32, device=self._device)

        if int(self._mjm.nv) >= 6:
            out["base_ang_vel_xyz"] = qvel[:, 3:6]
        else:
            out["base_ang_vel_xyz"] = torch.zeros((self._nworld, 3), dtype=torch.float32, device=self._device)

        return out

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False

    def set_joint_armature(self, armature: float | Iterable[float], joint_keys: Iterable[str] | None = None) -> None:
        self._require_connected()
        ids = list(range(len(JOINT_ORDER))) if joint_keys is None else [JOINT_ORDER.index(j) for j in joint_keys]
        vals = self._expand_param(armature, len(ids), "armature")
        for i, idx in enumerate(ids):
            self._mjm.dof_armature[self._joint_qvel_idx[idx]] = float(vals[i])
        self._sync_device_model()

    def set_joint_viscous_friction(self, viscous: float | Iterable[float], joint_keys: Iterable[str] | None = None) -> None:
        self._require_connected()
        ids = list(range(len(JOINT_ORDER))) if joint_keys is None else [JOINT_ORDER.index(j) for j in joint_keys]
        vals = self._expand_param(viscous, len(ids), "viscous")
        for i, idx in enumerate(ids):
            self._mjm.dof_damping[self._joint_qvel_idx[idx]] = float(vals[i])
        self._sync_device_model()

    def set_joint_friction(self, frictionloss: float | Iterable[float], joint_keys: Iterable[str] | None = None) -> None:
        self._require_connected()
        ids = list(range(len(JOINT_ORDER))) if joint_keys is None else [JOINT_ORDER.index(j) for j in joint_keys]
        vals = self._expand_param(frictionloss, len(ids), "frictionloss")
        for i, idx in enumerate(ids):
            self._mjm.dof_frictionloss[self._joint_qvel_idx[idx]] = float(vals[i])
        self._sync_device_model()

    def set_joint_torque_limit(self, torque_limit: float | Iterable[float], joint_keys: Iterable[str] | None = None) -> None:
        self._require_connected()
        ids = list(range(len(JOINT_ORDER))) if joint_keys is None else [JOINT_ORDER.index(j) for j in joint_keys]
        vals = self._expand_param(torque_limit, len(ids), "torque_limit")
        for i, idx in enumerate(ids):
            lim = float(vals[i])
            if not np.isfinite(lim) or lim <= 0.0:
                raise ValueError(f"torque_limit must be finite and > 0, got {lim}")
            aid = int(self._actuator_idx[idx])
            jid = int(self._joint_mj_idx[idx])
            # Keep limits symmetric around zero.
            self._mjm.actuator_forcerange[aid, 0] = -lim
            self._mjm.actuator_forcerange[aid, 1] = lim
            if hasattr(self._mjm, "actuator_forcelimited"):
                self._mjm.actuator_forcelimited[aid] = 1
            if hasattr(self._mjm, "jnt_actfrcrange"):
                self._mjm.jnt_actfrcrange[jid, 0] = -lim
                self._mjm.jnt_actfrcrange[jid, 1] = lim
            if hasattr(self._mjm, "jnt_actfrclimited"):
                self._mjm.jnt_actfrclimited[jid] = 1
        self._sync_device_model()

    def set_joint_action_delay_steps(
        self,
        action_delay_steps: float | Iterable[float],
        joint_keys: Iterable[str] | None = None,
    ) -> None:
        self._require_connected()
        ids = list(range(len(JOINT_ORDER))) if joint_keys is None else [JOINT_ORDER.index(j) for j in joint_keys]
        vals = self._expand_param(action_delay_steps, len(ids), "action_delay_steps")
        for i, idx in enumerate(ids):
            v = int(max(0, round(float(vals[i]))))
            self._action_delay_steps[idx] = v

    def set_actuator_gains(
        self,
        kp: float | Iterable[float] | None = None,
        kd: float | Iterable[float] | None = None,
        joint_keys: Iterable[str] | None = None,
    ) -> None:
        self._require_connected()
        ids = list(range(len(JOINT_ORDER))) if joint_keys is None else [JOINT_ORDER.index(j) for j in joint_keys]
        kp_vals = self._expand_param(kp, len(ids), "kp") if kp is not None else None
        kd_vals = self._expand_param(kd, len(ids), "kd") if kd is not None else None

        if kp_vals is not None:
            for i, idx in enumerate(ids):
                aid = self._actuator_idx[idx]
                kpv = float(kp_vals[i])
                self._mjm.actuator_gainprm[aid, 0] = kpv
                self._mjm.actuator_biasprm[aid, 1] = -kpv
        if kd_vals is not None:
            for i, idx in enumerate(ids):
                aid = self._actuator_idx[idx]
                self._mjm.actuator_biasprm[aid, 2] = -float(kd_vals[i])
        self._sync_device_model()

    def set_body_mass(self, mass: float | Iterable[float], body_names: Iterable[str]) -> None:
        self._require_connected()
        names = [str(n) for n in body_names]
        vals = self._expand_param(mass, len(names), "mass")
        for i, bname in enumerate(names):
            bid = self._body_name_to_idx.get(bname)
            if bid is None:
                raise ValueError(f"Unknown body name '{bname}'")
            self._mjm.body_mass[bid] = float(vals[i])
        self._sync_device_model()

    def set_body_com(self, com_xyz: Iterable[Iterable[float]], body_names: Iterable[str]) -> None:
        self._require_connected()
        names = [str(n) for n in body_names]
        com_arr = np.asarray(list(com_xyz), dtype=np.float64)
        if com_arr.shape != (len(names), 3):
            raise ValueError(f"Expected com_xyz shape {(len(names), 3)}, got {com_arr.shape}")
        for i, bname in enumerate(names):
            bid = self._body_name_to_idx.get(bname)
            if bid is None:
                raise ValueError(f"Unknown body name '{bname}'")
            self._mjm.body_ipos[bid, :] = com_arr[i, :]
        self._sync_device_model()

    @staticmethod
    def _expand_param(values: float | Iterable[float], n: int, name: str) -> list[float]:
        if isinstance(values, (float, int)):
            return [float(values)] * n
        out = [float(v) for v in values]
        if len(out) != n:
            raise ValueError(f"{name} length mismatch: expected {n}, got {len(out)}")
        return out


def _extract_actions_deg_from_dataset(dataset: Any) -> np.ndarray:
    # Supported dataset formats:
    # - ndarray/list with shape (T, 12)
    # - dict containing "actions_deg" or "action_deg"
    # - object with attribute "actions_deg" or "action_deg"
    if isinstance(dataset, dict):
        if "actions_deg" in dataset:
            arr = dataset["actions_deg"]
        elif "action_deg" in dataset:
            arr = dataset["action_deg"]
        else:
            raise ValueError("Dataset dict must contain 'actions_deg' or 'action_deg'.")
    elif hasattr(dataset, "actions_deg"):
        arr = getattr(dataset, "actions_deg")
    elif hasattr(dataset, "action_deg"):
        arr = getattr(dataset, "action_deg")
    else:
        arr = dataset
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != len(JOINT_ORDER):
        raise ValueError(
            f"Each dataset must provide actions with shape (T, {len(JOINT_ORDER)}), got {out.shape}"
        )
    return out


def _extract_timestamps_s_from_dataset(dataset: Any) -> np.ndarray | None:
    # Supported timestamp formats:
    # - dict containing "timestamps_s" or "timestamp_s"
    # - object with attribute "timestamps_s" or "timestamp_s"
    ts = None
    if isinstance(dataset, dict):
        if "timestamps_s" in dataset:
            ts = dataset["timestamps_s"]
        elif "timestamp_s" in dataset:
            ts = dataset["timestamp_s"]
    elif hasattr(dataset, "timestamps_s"):
        ts = getattr(dataset, "timestamps_s")
    elif hasattr(dataset, "timestamp_s"):
        ts = getattr(dataset, "timestamp_s")

    if ts is None:
        return None
    arr = np.asarray(ts, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        return None
    return arr


def _extract_observed_pos_deg_from_dataset(dataset: Any) -> np.ndarray | None:
    # Supported observed position formats:
    # - dict containing "observed_pos_deg"
    # - object with attribute "observed_pos_deg"
    obs = None
    if isinstance(dataset, dict):
        if "observed_pos_deg" in dataset:
            obs = dataset["observed_pos_deg"]
    elif hasattr(dataset, "observed_pos_deg"):
        obs = getattr(dataset, "observed_pos_deg")

    if obs is None:
        return None
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(JOINT_ORDER) or arr.shape[0] < 1:
        return None
    return arr


def _extract_observed_vel_deg_s_from_dataset(dataset: Any) -> np.ndarray | None:
    # Supported observed velocity formats:
    # - dict containing "observed_vel_deg_s"
    # - object with attribute "observed_vel_deg_s"
    obs = None
    if isinstance(dataset, dict):
        if "observed_vel_deg_s" in dataset:
            obs = dataset["observed_vel_deg_s"]
    elif hasattr(dataset, "observed_vel_deg_s"):
        obs = getattr(dataset, "observed_vel_deg_s")

    if obs is None:
        return None
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(JOINT_ORDER) or arr.shape[0] < 1:
        return None
    return arr


@dataclass
class _PreparedReplayBatch:
    nworld: int
    lengths: np.ndarray  # [nworld]
    tmax: int
    nrep: int
    valid_mask: np.ndarray  # [nworld, tmax]
    actions_deg_np_env_major: np.ndarray  # [nworld, tmax, Nj]
    actions_deg_t_time_major: Any  # torch [tmax, nworld, Nj]
    actions_rad_t_time_major: Any  # torch [tmax, nworld, Nj]
    init_pos_rad_t: Any  # torch [nworld, Nj]
    init_vel_rad_t: Any  # torch [nworld, Nj]
    pos_hist_t: Any  # torch [tmax, nworld, Nj]
    vel_hist_t: Any  # torch [tmax, nworld, Nj]


def _default_replay_cache_key(datasets: list[Any]) -> str:
    parts = [f"n={len(datasets)}"]
    for ds in datasets:
        root = str(getattr(ds, "dataset_root", ""))
        ep = getattr(ds, "episode_index", None)
        parts.append(f"id={id(ds)}|root={root}|ep={ep}")
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def _prepare_replay_batch_single_model(model: HumanoidMJWarp, datasets: list[Any]) -> _PreparedReplayBatch:
    nworld = int(model.num_worlds)
    if len(datasets) != nworld:
        raise ValueError(f"Expected {nworld} datasets (one per env), got {len(datasets)}")

    actions_by_env = [_extract_actions_deg_from_dataset(ds) for ds in datasets]
    ts_by_env = [_extract_timestamps_s_from_dataset(ds) for ds in datasets]
    observed_by_env = [_extract_observed_pos_deg_from_dataset(ds) for ds in datasets]
    observed_vel_by_env = [_extract_observed_vel_deg_s_from_dataset(ds) for ds in datasets]

    lengths = np.asarray([a.shape[0] for a in actions_by_env], dtype=np.int64)
    tmax = int(np.max(lengths)) if lengths.size > 0 else 0

    dt_candidates = []
    for ts in ts_by_env:
        if ts is None:
            continue
        d = np.diff(ts)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size > 0:
            dt_candidates.append(float(np.median(d)))
    if len(dt_candidates) > 0:
        dt_data = float(np.median(np.asarray(dt_candidates, dtype=np.float64)))
        nrep = int(max(1, round(dt_data / max(float(model._sim_dt_s), 1e-9))))
    else:
        nrep = 1

    actions_time_major = np.zeros((tmax, nworld, len(JOINT_ORDER)), dtype=np.float32)
    valid_mask = np.zeros((nworld, tmax), dtype=bool)
    init_deg = np.zeros((nworld, len(JOINT_ORDER)), dtype=np.float32)
    init_vel_deg_s = np.zeros((nworld, len(JOINT_ORDER)), dtype=np.float32)

    for wid, acts in enumerate(actions_by_env):
        L = int(acts.shape[0])
        if L > 0:
            actions_time_major[:L, wid, :] = acts
            if L < tmax:
                actions_time_major[L:, wid, :] = acts[-1, :]
            valid_mask[wid, :L] = True

        obs = observed_by_env[wid]
        obs_vel = observed_vel_by_env[wid]
        if obs is not None and obs.shape[0] > 0:
            init_deg[wid, :] = obs[0, :]
        elif L > 0:
            init_deg[wid, :] = acts[0, :]
        if obs_vel is not None and obs_vel.shape[0] > 0:
            init_vel_deg_s[wid, :] = obs_vel[0, :]

    init_rad = init_deg * (np.pi / 180.0)
    init_vel_rad_s = init_vel_deg_s * (np.pi / 180.0)

    torch = model._torch
    actions_deg_t_time_major = torch.as_tensor(actions_time_major, device=model._device, dtype=torch.float32)
    actions_rad_t_time_major = actions_deg_t_time_major * float(np.pi / 180.0)
    init_pos_rad_t = torch.as_tensor(init_rad, device=model._device, dtype=torch.float32)
    init_vel_rad_t = torch.as_tensor(init_vel_rad_s, device=model._device, dtype=torch.float32)
    pos_hist_t = torch.empty((tmax, nworld, len(JOINT_ORDER)), device=model._device, dtype=torch.float32)
    vel_hist_t = torch.empty((tmax, nworld, len(JOINT_ORDER)), device=model._device, dtype=torch.float32)

    return _PreparedReplayBatch(
        nworld=nworld,
        lengths=lengths,
        tmax=tmax,
        nrep=nrep,
        valid_mask=valid_mask,
        actions_deg_np_env_major=np.transpose(actions_time_major, (1, 0, 2)),
        actions_deg_t_time_major=actions_deg_t_time_major,
        actions_rad_t_time_major=actions_rad_t_time_major,
        init_pos_rad_t=init_pos_rad_t,
        init_vel_rad_t=init_vel_rad_t,
        pos_hist_t=pos_hist_t,
        vel_hist_t=vel_hist_t,
    )


def _apply_replay_initial_state(model: HumanoidMJWarp, prepared: _PreparedReplayBatch) -> None:
    if prepared.tmax <= 0:
        return
    qpos = model._wp.to_torch(model._d.qpos)
    qvel = model._wp.to_torch(model._d.qvel)
    qpos.index_copy_(1, model._joint_qpos_idx_t, prepared.init_pos_rad_t)
    qvel.index_copy_(1, model._joint_qvel_idx_t, prepared.init_vel_rad_t)
    model._d.qpos = model._wp.from_torch(qpos, dtype=model._wp.float32)
    model._d.qvel = model._wp.from_torch(qvel, dtype=model._wp.float32)


def _replay_prepared_batch_single_model(
    model: HumanoidMJWarp,
    prepared: _PreparedReplayBatch,
    *,
    reset_before: bool = True,
    initialize_from_first_action: bool = True,
    include_action: bool = True,
    include_velocity: bool = True,
) -> dict[str, np.ndarray]:
    if reset_before:
        model.reset()
    if initialize_from_first_action:
        _apply_replay_initial_state(model, prepared)

    torch = model._torch
    wp = model._wp
    mjw = model._mjw
    m = model._m
    d = model._d
    ctrl = model._ctrl
    actuator_idx_t = model._actuator_idx_t
    joint_qpos_idx_t = model._joint_qpos_idx_t
    joint_qvel_idx_t = model._joint_qvel_idx_t
    rad_to_deg = float(180.0 / np.pi)

    tmax = int(prepared.tmax)
    nrep = int(prepared.nrep)
    delay_steps_np = np.asarray(getattr(model, "_action_delay_steps", np.zeros((len(JOINT_ORDER),), dtype=np.int64)), dtype=np.int64).reshape(-1)
    if delay_steps_np.shape != (len(JOINT_ORDER),):
        raise RuntimeError(
            f"Invalid model action delay shape: {delay_steps_np.shape}, expected {(len(JOINT_ORDER),)}"
        )
    if np.any(delay_steps_np < 0):
        raise RuntimeError(f"Invalid negative action delay steps: {delay_steps_np}")
    use_delay_queue = bool(np.any(delay_steps_np > 0)) and tmax > 0
    delay_queues: list[deque[Any] | None] = []
    if use_delay_queue:
        first_cmd = prepared.actions_rad_t_time_major[0]
        for jidx, d_steps in enumerate(delay_steps_np.tolist()):
            d_int = int(d_steps)
            if d_int <= 0:
                delay_queues.append(None)
                continue
            init_col = first_cmd[:, jidx].clone()
            delay_queues.append(deque((init_col.clone() for _ in range(d_int)), maxlen=d_int))
    start_sim_time_s = float(model.get_sim_time_s())
    for t in range(tmax):
        # Fast-path replay: avoid per-step wrapper calls and convert-to-rad only once.
        if use_delay_queue:
            cmd_t = prepared.actions_rad_t_time_major[t]
            delayed_cmd_t = cmd_t.clone()
            for jidx, q in enumerate(delay_queues):
                if q is None:
                    continue
                delayed_cmd_t[:, jidx] = q.popleft()
                q.append(cmd_t[:, jidx].clone())
            ctrl[:, actuator_idx_t] = delayed_cmd_t
        else:
            ctrl[:, actuator_idx_t] = prepared.actions_rad_t_time_major[t]
        d.ctrl = wp.from_torch(ctrl, dtype=wp.float32)
        for _ in range(nrep):
            mjw.step(m, d)

        qpos_t = wp.to_torch(d.qpos)
        prepared.pos_hist_t[t, :, :] = torch.index_select(qpos_t, dim=1, index=joint_qpos_idx_t) * rad_to_deg
        if include_velocity:
            qvel_t = wp.to_torch(d.qvel)
            prepared.vel_hist_t[t, :, :] = torch.index_select(qvel_t, dim=1, index=joint_qvel_idx_t) * rad_to_deg

    model._sim_time_s = start_sim_time_s + float(tmax) * float(model._sim_dt_s) * float(nrep)

    sim_time_s = start_sim_time_s + (
        np.arange(1, tmax + 1, dtype=np.float64) * float(model._sim_dt_s) * float(nrep)
    )

    out: dict[str, np.ndarray] = {
        "joint_pos_deg": prepared.pos_hist_t.permute(1, 0, 2).detach().cpu().numpy().astype(np.float32, copy=False),
        "sim_time_s": sim_time_s,
        "valid_mask": prepared.valid_mask,
        "lengths": prepared.lengths,
        "action_repeat_used": int(prepared.nrep),
    }
    if include_velocity:
        out["joint_vel_deg_s"] = (
            prepared.vel_hist_t.permute(1, 0, 2).detach().cpu().numpy().astype(np.float32, copy=False)
        )
    if include_action:
        out["action_deg"] = prepared.actions_deg_np_env_major
    return out


def _replay_datasets_actions_single_model(
    model: HumanoidMJWarp,
    datasets: list[Any],
    *,
    reset_before: bool = True,
    initialize_from_first_action: bool = True,
    include_action: bool = True,
    include_velocity: bool = True,
) -> dict[str, np.ndarray]:
    prepared = _prepare_replay_batch_single_model(model, datasets)
    return _replay_prepared_batch_single_model(
        model,
        prepared,
        reset_before=reset_before,
        initialize_from_first_action=initialize_from_first_action,
        include_action=include_action,
        include_velocity=include_velocity,
    )


def _worker_readonly_snapshot(model: HumanoidMJWarp) -> dict[str, Any]:
    mj = model._mujoco
    torque = [_actuator_torque_limit_or_default(model._mjm, aid) for aid in model._actuator_idx]
    out: dict[str, Any] = {
        "armature": np.asarray([float(model._mjm.dof_armature[d]) for d in model._joint_qvel_idx], dtype=np.float64),
        "viscous": np.asarray([float(model._mjm.dof_damping[d]) for d in model._joint_qvel_idx], dtype=np.float64),
        "dry": np.asarray([float(model._mjm.dof_frictionloss[d]) for d in model._joint_qvel_idx], dtype=np.float64),
        "torque_limit": np.asarray(torque, dtype=np.float64),
        "action_delay_steps": np.asarray(model._action_delay_steps, dtype=np.float64),
        "kp": np.asarray([float(model._mjm.actuator_gainprm[a, 0]) for a in model._actuator_idx], dtype=np.float64),
        "kv": np.asarray([float(-model._mjm.actuator_biasprm[a, 2]) for a in model._actuator_idx], dtype=np.float64),
        "body_name_by_joint": {},
        "body_mass_by_joint": {},
        "body_com_by_joint": {},
    }
    for j in JOINT_ORDER:
        mj_joint_name = _MJ_JOINT_NAME_BY_KEY[j]
        jid = int(mj.mj_name2id(model._mjm, mj.mjtObj.mjOBJ_JOINT, mj_joint_name))
        if jid < 0:
            continue
        bid = int(model._mjm.jnt_bodyid[jid])
        bname = mj.mj_id2name(model._mjm, mj.mjtObj.mjOBJ_BODY, bid)
        if not bname:
            continue
        out["body_name_by_joint"][j] = str(bname)
        out["body_mass_by_joint"][j] = float(model._mjm.body_mass[bid])
        out["body_com_by_joint"][j] = np.asarray(model._mjm.body_ipos[bid], dtype=np.float64).copy()
    return out


def _model_worker_loop(
    conn: Any,
    config: HumanoidMJWarpConfig,
    worker_index: int,
    cpu_id: int | None,
    worker_num_threads: int,
) -> None:
    # Limit CPU thread fan-out inside each worker process to avoid oversubscription.
    nthread = int(max(0, worker_num_threads))
    if nthread > 0:
        v = str(nthread)
        os.environ["OMP_NUM_THREADS"] = v
        os.environ["MKL_NUM_THREADS"] = v
        os.environ["OPENBLAS_NUM_THREADS"] = v
        os.environ["NUMEXPR_NUM_THREADS"] = v
        os.environ["VECLIB_MAXIMUM_THREADS"] = v

    pinned_cpu: int | None = None
    if cpu_id is not None:
        try:
            c = int(cpu_id)
            os.sched_setaffinity(0, {c})
            pinned_cpu = c
        except Exception:
            pinned_cpu = None

    model = HumanoidMJWarp(config)
    if nthread > 0:
        try:
            model._torch.set_num_threads(int(max(1, nthread)))
            model._torch.set_num_interop_threads(1)
        except Exception:
            pass
    model.connect()
    replay_cache: dict[str, _PreparedReplayBatch] = {}
    conn.send(
        {
            "ok": True,
            "worker_index": int(worker_index),
            "cpu_id": pinned_cpu,
            "worker_num_threads": int(nthread),
        }
    )
    while True:
        msg = conn.recv()
        cmd = msg.get("cmd")
        payload = msg.get("payload", {})
        try:
            if cmd == "disconnect":
                if model.is_connected:
                    model.disconnect()
                conn.send({"ok": True})
                break
            if cmd == "reset":
                model.reset()
                conn.send({"ok": True})
                continue
            if cmd == "step":
                model.step(nstep=int(payload["nstep"]))
                conn.send({"ok": True})
                continue
            if cmd == "send_action_tensor":
                model.send_action_tensor(payload["action_deg"], step=bool(payload.get("step", True)))
                conn.send({"ok": True})
                continue
            if cmd == "get_observation_tensor":
                obs = model.get_observation_tensor()
                conn.send(
                    {
                        "ok": True,
                        "obs": {
                            "joint_pos_deg": obs["joint_pos_deg"].detach().cpu().numpy().astype(np.float32, copy=False),
                            "joint_vel_deg_s": obs["joint_vel_deg_s"].detach().cpu().numpy().astype(np.float32, copy=False),
                            "base_orientation_xyzw": obs["base_orientation_xyzw"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False),
                            "base_ang_vel_xyz": obs["base_ang_vel_xyz"].detach().cpu().numpy().astype(np.float32, copy=False),
                        },
                    }
                )
                continue
            if cmd == "set_joint_armature":
                model.set_joint_armature(armature=payload["armature"], joint_keys=payload.get("joint_keys"))
                conn.send({"ok": True})
                continue
            if cmd == "set_joint_viscous":
                model.set_joint_viscous_friction(viscous=payload["viscous"], joint_keys=payload.get("joint_keys"))
                conn.send({"ok": True})
                continue
            if cmd == "set_joint_static":
                model.set_joint_friction(frictionloss=payload["frictionloss"], joint_keys=payload.get("joint_keys"))
                conn.send({"ok": True})
                continue
            if cmd == "set_joint_torque_limit":
                model.set_joint_torque_limit(torque_limit=payload["torque_limit"], joint_keys=payload.get("joint_keys"))
                conn.send({"ok": True})
                continue
            if cmd == "set_joint_action_delay_steps":
                model.set_joint_action_delay_steps(
                    action_delay_steps=payload["action_delay_steps"],
                    joint_keys=payload.get("joint_keys"),
                )
                conn.send({"ok": True})
                continue
            if cmd == "set_joint_gains":
                model.set_actuator_gains(kp=payload.get("kp"), kd=payload.get("kv"), joint_keys=payload.get("joint_keys"))
                conn.send({"ok": True})
                continue
            if cmd == "set_body_mass":
                model.set_body_mass(mass=payload["mass"], body_names=payload["body_names"])
                conn.send({"ok": True})
                continue
            if cmd == "set_body_com":
                model.set_body_com(com_xyz=payload["com_xyz"], body_names=payload["body_names"])
                conn.send({"ok": True})
                continue
            if cmd == "prepare_replay":
                cache_key = str(payload["cache_key"])
                replay_cache[cache_key] = _prepare_replay_batch_single_model(model, payload["datasets"])
                conn.send({"ok": True, "cache_key": cache_key})
                continue
            if cmd == "replay_prepared":
                cache_key = str(payload["cache_key"])
                prepared = replay_cache.get(cache_key)
                if prepared is None:
                    raise RuntimeError(f"Unknown replay cache key '{cache_key}'")
                replay = _replay_prepared_batch_single_model(
                    model,
                    prepared,
                    reset_before=bool(payload.get("reset_before", True)),
                    initialize_from_first_action=bool(payload.get("initialize_from_first_action", True)),
                    include_action=bool(payload.get("include_action", True)),
                    include_velocity=bool(payload.get("include_velocity", True)),
                )
                conn.send({"ok": True, "replay": replay})
                continue
            if cmd == "replay":
                replay = _replay_datasets_actions_single_model(
                    model,
                    payload["datasets"],
                    reset_before=bool(payload.get("reset_before", True)),
                    initialize_from_first_action=bool(payload.get("initialize_from_first_action", True)),
                    include_action=bool(payload.get("include_action", True)),
                    include_velocity=bool(payload.get("include_velocity", True)),
                )
                conn.send({"ok": True, "replay": replay})
                continue
            if cmd == "snapshot":
                conn.send({"ok": True, "snapshot": _worker_readonly_snapshot(model)})
                continue
            if cmd == "runtime_joint_params":
                jidx = int(payload["joint_idx"])
                qd = model._joint_qvel_idx[jidx]
                aid = model._actuator_idx[jidx]
                conn.send(
                    {
                        "ok": True,
                        "params": {
                            "armature": float(model._mjm.dof_armature[qd]),
                            "viscous": float(model._mjm.dof_damping[qd]),
                            "dry": float(model._mjm.dof_frictionloss[qd]),
                            "torque_limit": _actuator_torque_limit_or_default(model._mjm, aid),
                            "action_delay_steps": int(model._action_delay_steps[jidx]),
                            "kp": float(model._mjm.actuator_gainprm[aid, 0]),
                            "kv": float(-model._mjm.actuator_biasprm[aid, 2]),
                        },
                    }
                )
                continue
            conn.send({"ok": False, "error": f"Unknown worker command: {cmd}"})
        except Exception as exc:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class HumanoidMJWarpModelPool:
    """
    First-layer parallel abstraction:
    - n_model independent MJWarp models
    - each model runs nworld environments in parallel
    """

    def __init__(
        self,
        config: HumanoidMJWarpConfig,
        n_model: int,
        *,
        use_multiprocessing: bool = False,
        mp_start_method: str = "spawn",
        pin_workers_to_cores: bool = False,
        prefer_physical_cores: bool = True,
        worker_num_threads: int = 1,
    ):
        self.n_model = int(max(1, n_model))
        self.nworld = int(max(1, config.nworld))
        self._base_config = replace(config)
        self._use_multiprocessing = bool(use_multiprocessing)
        self._mp_start_method = str(mp_start_method)
        self._pin_workers_to_cores = bool(pin_workers_to_cores)
        self._prefer_physical_cores = bool(prefer_physical_cores)
        self._worker_num_threads = int(max(0, worker_num_threads))
        base_id = str(config.id) if config.id is not None else "humanoid_mjwarp"
        self._model_cfgs: list[HumanoidMJWarpConfig] = []
        for i in range(self.n_model):
            cfg_i = replace(config)
            cfg_i.id = f"{base_id}_model{i:03d}"
            self._model_cfgs.append(cfg_i)
        self._models: list[HumanoidMJWarp] = []
        self._workers: list[tuple[Any, mp.Process]] = []
        self._worker_cpu_ids: list[int | None] = []
        if not self._use_multiprocessing:
            self._models = [HumanoidMJWarp(cfg) for cfg in self._model_cfgs]
        self._sim_time_s = 0.0
        self._prepared_replay_keys: set[str] = set()

    @property
    def models(self) -> list[HumanoidMJWarp]:
        if self._use_multiprocessing:
            raise RuntimeError("`models` is unavailable in multiprocessing mode. Use snapshot/runtime getters.")
        return self._models

    @property
    def worker_cpu_ids(self) -> list[int | None]:
        return list(self._worker_cpu_ids)

    @property
    def is_connected(self) -> bool:
        if self._use_multiprocessing:
            return len(self._workers) == self.n_model and all(proc.is_alive() for _, proc in self._workers)
        return all(m.is_connected for m in self._models)

    def _rpc_one(self, mid: int, cmd: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        conn, proc = self._workers[mid]
        if not proc.is_alive():
            raise RuntimeError(f"Worker process {mid} is not alive.")
        conn.send({"cmd": cmd, "payload": payload or {}})
        reply = conn.recv()
        if not bool(reply.get("ok", False)):
            raise RuntimeError(f"Worker {mid} command '{cmd}' failed: {reply.get('error', 'unknown error')}")
        return reply

    def _rpc_all(self, cmd: str, payload: dict[str, Any] | None = None, payload_by_model: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if payload_by_model is not None and len(payload_by_model) != self.n_model:
            raise ValueError(f"payload_by_model must have {self.n_model} items, got {len(payload_by_model)}")
        # Broadcast first so workers execute in parallel, then collect all replies.
        for mid, (conn, proc) in enumerate(self._workers):
            if not proc.is_alive():
                raise RuntimeError(f"Worker process {mid} is not alive.")
            p = payload_by_model[mid] if payload_by_model is not None else (payload or {})
            conn.send({"cmd": cmd, "payload": p})
        replies: list[dict[str, Any]] = []
        for mid in range(self.n_model):
            conn, _ = self._workers[mid]
            reply = conn.recv()
            if not bool(reply.get("ok", False)):
                raise RuntimeError(f"Worker {mid} command '{cmd}' failed: {reply.get('error', 'unknown error')}")
            replies.append(reply)
        return replies

    def connect(self) -> None:
        if self._use_multiprocessing:
            if self.is_connected:
                self._sim_time_s = 0.0
                return
            ctx = mp.get_context(self._mp_start_method)
            self._workers = []
            desired_cpu_ids = _pick_worker_cpu_ids(
                n_worker=self.n_model,
                pin_workers_to_cores=self._pin_workers_to_cores,
                prefer_physical_cores=self._prefer_physical_cores,
            )
            self._worker_cpu_ids = []
            for wid, cfg in enumerate(self._model_cfgs):
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(
                    target=_model_worker_loop,
                    args=(child_conn, cfg, int(wid), desired_cpu_ids[wid], int(self._worker_num_threads)),
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                ready = parent_conn.recv()
                if not bool(ready.get("ok", False)):
                    raise RuntimeError(f"Worker startup failed: {ready.get('error', 'unknown error')}")
                self._worker_cpu_ids.append(ready.get("cpu_id", None))
                self._workers.append((parent_conn, proc))
        else:
            for m in self._models:
                if not m.is_connected:
                    m.connect()
        self._sim_time_s = 0.0

    def disconnect(self) -> None:
        if self._use_multiprocessing:
            if not self._workers:
                return
            for mid in range(len(self._workers)):
                conn, proc = self._workers[mid]
                if proc.is_alive():
                    try:
                        conn.send({"cmd": "disconnect", "payload": {}})
                    except Exception:
                        pass
            for conn, proc in self._workers:
                if proc.is_alive():
                    try:
                        _ = conn.recv()
                    except Exception:
                        pass
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.terminate()
                try:
                    conn.close()
                except Exception:
                    pass
            self._workers = []
            self._worker_cpu_ids = []
        else:
            for m in self._models:
                if m.is_connected:
                    m.disconnect()
                m._replay_cache.clear()
        self._prepared_replay_keys.clear()

    def reset(self) -> None:
        if self._use_multiprocessing:
            self._rpc_all("reset")
        else:
            for m in self._models:
                m.reset()
        self._sim_time_s = 0.0

    def prepare_replay_datasets(self, datasets: list[Any], *, cache_key: str | None = None) -> str:
        if len(datasets) != self.nworld:
            raise ValueError(f"Expected {self.nworld} datasets (one per env), got {len(datasets)}")
        if not self.is_connected:
            raise RuntimeError("Pool is not connected. Call connect() first.")

        key = str(cache_key) if cache_key is not None else _default_replay_cache_key(datasets)
        if self._use_multiprocessing:
            if key not in self._prepared_replay_keys:
                self._rpc_all("prepare_replay", payload={"cache_key": key, "datasets": datasets})
                self._prepared_replay_keys.add(key)
            return key

        for m in self._models:
            if key not in m._replay_cache:
                m._replay_cache[key] = _prepare_replay_batch_single_model(m, datasets)
        self._prepared_replay_keys.add(key)
        return key

    def step(self, nstep: int = 1) -> None:
        n = int(max(1, nstep))
        if self._use_multiprocessing:
            self._rpc_all("step", payload={"nstep": n})
            dt = float(self._base_config.sim_dt)
        else:
            for m in self._models:
                m.step(nstep=n)
            dt = float(self._models[0]._sim_dt_s)
        self._sim_time_s += dt * float(n)

    def get_sim_time_s(self) -> float:
        return float(self._sim_time_s)

    def send_action_tensor(self, action_deg: Any, step: bool = True) -> Any:
        """
        action_deg shapes accepted:
        - (n_model, nworld, 12): per-model, per-env actions
        - (n_model, 12): per-model actions replicated across envs
        """
        arr = np.asarray(action_deg, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (self.n_model, len(JOINT_ORDER)):
            arr = np.repeat(arr[:, None, :], self.nworld, axis=1)
        if arr.ndim != 3 or arr.shape != (self.n_model, self.nworld, len(JOINT_ORDER)):
            raise ValueError(
                f"Expected action shape {(self.n_model, self.nworld, len(JOINT_ORDER))} "
                f"or {(self.n_model, len(JOINT_ORDER))}, got {arr.shape}"
            )
        if self._use_multiprocessing:
            payloads = [{"action_deg": arr[i], "step": False} for i in range(self.n_model)]
            self._rpc_all("send_action_tensor", payload_by_model=payloads)
        else:
            for i, m in enumerate(self._models):
                m.send_action_tensor(arr[i], step=False)
        if step:
            self.step(nstep=1)
        return arr

    def get_observation_tensor(self) -> dict[str, Any]:
        """
        Returns tensor dict with a leading model axis:
        key tensor shape: (n_model, nworld, ...)
        """
        if self._use_multiprocessing:
            replies = self._rpc_all("get_observation_tensor")
            out: dict[str, Any] = {}
            first = replies[0]["obs"]
            for k in first.keys():
                out[k] = np.stack([np.asarray(rep["obs"][k]) for rep in replies], axis=0)
            return out
        obs_by_model = [m.get_observation_tensor() for m in self._models]
        torch = self._models[0]._torch
        out: dict[str, Any] = {}
        for k in obs_by_model[0].keys():
            out[k] = torch.stack([obs[k] for obs in obs_by_model], dim=0)
        return out

    def get_model_snapshot(self, model_index: int = 0) -> dict[str, Any]:
        mid = int(model_index)
        if mid < 0 or mid >= self.n_model:
            raise ValueError(f"model_index out of bounds: {mid}")
        if self._use_multiprocessing:
            return self._rpc_one(mid, "snapshot")["snapshot"]
        return _worker_readonly_snapshot(self._models[mid])

    def get_runtime_joint_params(self, *, joint: str, model_indices: Iterable[int] | None = None) -> list[dict[str, float]]:
        if joint not in JOINT_INDEX:
            raise ValueError(f"Unknown joint '{joint}'")
        mids = list(range(self.n_model)) if model_indices is None else [int(i) for i in model_indices]
        jidx = JOINT_INDEX[joint]
        out: list[dict[str, float]] = []
        if self._use_multiprocessing:
            for mid in mids:
                out.append(self._rpc_one(mid, "runtime_joint_params", {"joint_idx": jidx})["params"])
            return out
        for mid in mids:
            m = self._models[mid]
            qd = m._joint_qvel_idx[jidx]
            aid = m._actuator_idx[jidx]
            out.append(
                {
                    "armature": float(m._mjm.dof_armature[qd]),
                    "viscous": float(m._mjm.dof_damping[qd]),
                    "dry": float(m._mjm.dof_frictionloss[qd]),
                    "torque_limit": _actuator_torque_limit_or_default(m._mjm, aid),
                    "action_delay_steps": int(m._action_delay_steps[jidx]),
                    "kp": float(m._mjm.actuator_gainprm[aid, 0]),
                    "kv": float(-m._mjm.actuator_biasprm[aid, 2]),
                }
            )
        return out

    @staticmethod
    def _as_model_matrix(values: Any, n_model: int, width: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full((n_model, width), float(arr), dtype=np.float64)
        elif arr.ndim == 1:
            if arr.size == width:
                arr = np.repeat(arr[None, :], n_model, axis=0)
            elif arr.size == n_model and width == 1:
                arr = arr.reshape(n_model, 1)
            else:
                raise ValueError(f"{name}: expected shape ({width},) or ({n_model},{width}), got {arr.shape}")
        elif arr.ndim == 2 and arr.shape == (n_model, width):
            pass
        else:
            raise ValueError(f"{name}: expected shape ({n_model},{width}), got {arr.shape}")
        return arr

    def set_joint_gains_per_model(
        self,
        *,
        kp: Any | None = None,
        kv: Any | None = None,
        joint_keys: Iterable[str] | None = None,
    ) -> None:
        if joint_keys is None:
            keys = list(JOINT_ORDER)
        else:
            keys = [str(j) for j in joint_keys]
        width = len(keys)
        kp_mat = None if kp is None else self._as_model_matrix(kp, self.n_model, width, "kp")
        kv_mat = None if kv is None else self._as_model_matrix(kv, self.n_model, width, "kv")
        if self._use_multiprocessing:
            payloads = []
            for i in range(self.n_model):
                payloads.append(
                    {
                        "kp": None if kp_mat is None else kp_mat[i, :].tolist(),
                        "kv": None if kv_mat is None else kv_mat[i, :].tolist(),
                        "joint_keys": keys,
                    }
                )
            self._rpc_all("set_joint_gains", payload_by_model=payloads)
        else:
            for i, m in enumerate(self._models):
                kp_i = None if kp_mat is None else kp_mat[i, :].tolist()
                kv_i = None if kv_mat is None else kv_mat[i, :].tolist()
                m.set_actuator_gains(kp=kp_i, kd=kv_i, joint_keys=keys)

    def set_joint_armature_per_model(self, armature: Any, joint_keys: Iterable[str] | None = None) -> None:
        keys = list(JOINT_ORDER) if joint_keys is None else [str(j) for j in joint_keys]
        mat = self._as_model_matrix(armature, self.n_model, len(keys), "armature")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_joint_armature",
                payload_by_model=[{"armature": mat[i, :].tolist(), "joint_keys": keys} for i in range(self.n_model)],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_joint_armature(armature=mat[i, :].tolist(), joint_keys=keys)

    def set_joint_viscous_friction_per_model(self, viscous: Any, joint_keys: Iterable[str] | None = None) -> None:
        keys = list(JOINT_ORDER) if joint_keys is None else [str(j) for j in joint_keys]
        mat = self._as_model_matrix(viscous, self.n_model, len(keys), "viscous")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_joint_viscous",
                payload_by_model=[{"viscous": mat[i, :].tolist(), "joint_keys": keys} for i in range(self.n_model)],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_joint_viscous_friction(viscous=mat[i, :].tolist(), joint_keys=keys)

    def set_joint_static_friction_per_model(self, frictionloss: Any, joint_keys: Iterable[str] | None = None) -> None:
        keys = list(JOINT_ORDER) if joint_keys is None else [str(j) for j in joint_keys]
        mat = self._as_model_matrix(frictionloss, self.n_model, len(keys), "frictionloss")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_joint_static",
                payload_by_model=[{"frictionloss": mat[i, :].tolist(), "joint_keys": keys} for i in range(self.n_model)],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_joint_friction(frictionloss=mat[i, :].tolist(), joint_keys=keys)

    def set_joint_torque_limit_per_model(self, torque_limit: Any, joint_keys: Iterable[str] | None = None) -> None:
        keys = list(JOINT_ORDER) if joint_keys is None else [str(j) for j in joint_keys]
        mat = self._as_model_matrix(torque_limit, self.n_model, len(keys), "torque_limit")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_joint_torque_limit",
                payload_by_model=[{"torque_limit": mat[i, :].tolist(), "joint_keys": keys} for i in range(self.n_model)],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_joint_torque_limit(torque_limit=mat[i, :].tolist(), joint_keys=keys)

    def set_joint_action_delay_steps_per_model(self, action_delay_steps: Any, joint_keys: Iterable[str] | None = None) -> None:
        keys = list(JOINT_ORDER) if joint_keys is None else [str(j) for j in joint_keys]
        mat = self._as_model_matrix(action_delay_steps, self.n_model, len(keys), "action_delay_steps")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_joint_action_delay_steps",
                payload_by_model=[
                    {"action_delay_steps": mat[i, :].tolist(), "joint_keys": keys}
                    for i in range(self.n_model)
                ],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_joint_action_delay_steps(action_delay_steps=mat[i, :].tolist(), joint_keys=keys)

    def set_body_mass_per_model(self, mass: Any, body_names: Iterable[str]) -> None:
        names = [str(b) for b in body_names]
        mat = self._as_model_matrix(mass, self.n_model, len(names), "mass")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_body_mass",
                payload_by_model=[{"mass": mat[i, :].tolist(), "body_names": names} for i in range(self.n_model)],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_body_mass(mass=mat[i, :].tolist(), body_names=names)

    def set_body_com_per_model(self, com_xyz: Any, body_names: Iterable[str]) -> None:
        names = [str(b) for b in body_names]
        arr = np.asarray(com_xyz, dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (len(names), 3):
            arr = np.repeat(arr[None, :, :], self.n_model, axis=0)
        if arr.shape != (self.n_model, len(names), 3):
            raise ValueError(f"com_xyz: expected shape {(self.n_model, len(names), 3)}, got {arr.shape}")
        if self._use_multiprocessing:
            self._rpc_all(
                "set_body_com",
                payload_by_model=[{"com_xyz": arr[i], "body_names": names} for i in range(self.n_model)],
            )
        else:
            for i, m in enumerate(self._models):
                m.set_body_com(com_xyz=arr[i], body_names=names)

    def replay_datasets_actions(
        self,
        datasets: list[Any],
        *,
        cache_key: str | None = None,
        reset_before: bool = True,
        initialize_from_first_action: bool = True,
        include_action: bool = True,
        include_velocity: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Replay dataset actions on each sub-env for all models in parallel.
        - len(datasets) must equal nworld.
        - dataset i drives sub-env i for every model.

        Returns:
            {
              "joint_pos_deg": (n_model, nworld, T, 12),
              "joint_vel_deg_s": (n_model, nworld, T, 12),
              "sim_time_s": (T,),
              "valid_mask": (nworld, T),
              "lengths": (nworld,),
            }
        """
        key = self.prepare_replay_datasets(datasets, cache_key=cache_key)
        if self._use_multiprocessing:
            payload = {
                "cache_key": key,
                "reset_before": bool(reset_before),
                "initialize_from_first_action": bool(initialize_from_first_action),
                "include_action": bool(include_action),
                "include_velocity": bool(include_velocity),
            }
            replies = self._rpc_all("replay_prepared", payload=payload)
            per_model = [rep["replay"] for rep in replies]
        else:
            per_model = [
                _replay_prepared_batch_single_model(
                    m,
                    m._replay_cache[key],
                    reset_before=reset_before,
                    initialize_from_first_action=initialize_from_first_action,
                    include_action=include_action,
                    include_velocity=include_velocity,
                )
                for m in self._models
            ]

        pos_hist = np.stack([np.asarray(r["joint_pos_deg"], dtype=np.float32) for r in per_model], axis=0)
        sim_time_s = np.asarray(per_model[0]["sim_time_s"], dtype=np.float64)
        valid_mask = np.asarray(per_model[0]["valid_mask"], dtype=bool)
        lengths = np.asarray(per_model[0]["lengths"], dtype=np.int64)
        action_repeat_used = int(per_model[0]["action_repeat_used"])
        self._sim_time_s = float(sim_time_s[-1]) if sim_time_s.size > 0 else 0.0
        out = {
            "joint_pos_deg": pos_hist,
            "sim_time_s": sim_time_s,
            "valid_mask": valid_mask,
            "lengths": lengths,
            "action_repeat_used": action_repeat_used,
        }
        if include_velocity:
            out["joint_vel_deg_s"] = np.stack([np.asarray(r["joint_vel_deg_s"], dtype=np.float32) for r in per_model], axis=0)
        if include_action:
            out["action_deg"] = np.stack([np.asarray(r["action_deg"], dtype=np.float32) for r in per_model], axis=0)
        return out
