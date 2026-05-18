from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from cmaes.dataset_loader import (
    JointMotionDataset,
)
from simulator.config import JOINT_ORDER
from simulator.runtime import HumanoidMJWarpModelPool

CANONICAL_PARAM_ORDER = (
    "armature",
    "viscous_friction",
    "dry_friction",
    "torque_limit",
    "action_delay_steps",
    "kp",
    "kv",
    "mass_scale",
    "com_x_scale",
    "com_y_scale",
    "com_z_scale",
)

PARAM_ALIASES = {
    "armature": "armature",
    "viscous": "viscous_friction",
    "viscous_friction": "viscous_friction",
    "damping": "viscous_friction",
    "dry": "dry_friction",
    "dry_friction": "dry_friction",
    "frictionloss": "dry_friction",
    "torque": "torque_limit",
    "torque_limit": "torque_limit",
    "force_limit": "torque_limit",
    "torque_limit_scale": "torque_limit_scale",
    "force_limit_scale": "torque_limit_scale",
    "delay": "action_delay_steps",
    "delay_steps": "action_delay_steps",
    "action_delay": "action_delay_steps",
    "action_delay_steps": "action_delay_steps",
    "kp": "kp",
    "kd": "kv",
    "kv": "kv",
    "mass": "mass",
    "body_mass": "mass",
    "mass_scale": "mass_scale",
    "mass_ratio": "mass_scale",
    "com_x": "com_x",
    "com_y": "com_y",
    "com_z": "com_z",
    "body_com_x": "com_x",
    "body_com_y": "com_y",
    "body_com_z": "com_z",
    "com_x_scale": "com_x_scale",
    "com_y_scale": "com_y_scale",
    "com_z_scale": "com_z_scale",
    "com_scale_x": "com_x_scale",
    "com_scale_y": "com_y_scale",
    "com_scale_z": "com_z_scale",
}

def _normalize_param_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in PARAM_ALIASES:
        raise ValueError(f"Unknown param '{name}'. Supported aliases: {sorted(PARAM_ALIASES.keys())}")
    return PARAM_ALIASES[key]


@dataclass(frozen=True)
class JointCandidateLayout:
    param_order: tuple[str, ...]

    @property
    def nparam(self) -> int:
        return len(self.param_order)


class MJWarpPopulationJointCost:
    """
    Population evaluator for one-joint-at-a-time identification.

    - n_model = pool population size
    - nworld = number of datasets replayed in parallel for the selected joint

    Cost per model:
      sum_over_worlds sum_over_time (q_sim - q_obs)^2
    """

    def __init__(
        self,
        *,
        pool: HumanoidMJWarpModelPool,
        datasets_by_joint: Mapping[str, Sequence[JointMotionDataset]],
        base_kp: Sequence[float],
        base_kv: Sequence[float],
        joint_order: Sequence[str] | None = None,
        param_order: Sequence[str] = CANONICAL_PARAM_ORDER,
    ):
        self.pool = pool
        if not self.pool.is_connected:
            self.pool.connect()
        self.n_model = int(self.pool.n_model)

        self.joint_order = list(joint_order) if joint_order is not None else list(JOINT_ORDER)
        self._joint_index = {j: i for i, j in enumerate(self.joint_order)}
        self._model_joint_index = {j: i for i, j in enumerate(JOINT_ORDER)}
        if len(self._joint_index) != len(self.joint_order):
            raise ValueError("joint_order contains duplicates.")
        if any(j not in self._model_joint_index for j in self.joint_order):
            missing = [j for j in self.joint_order if j not in self._model_joint_index]
            raise ValueError(f"Unknown joint(s) in joint_order: {missing}")

        norm_order = tuple(_normalize_param_name(p) for p in param_order)
        if len(set(norm_order)) != len(norm_order):
            raise ValueError(f"param_order has duplicates after normalization: {norm_order}")
        self.layout = JointCandidateLayout(param_order=norm_order)

        self._trajs_by_joint = self._load_trajectories_by_joint(datasets_by_joint)
        any_joint = next(iter(self._trajs_by_joint.keys()))
        self.nworld = int(len(self._trajs_by_joint[any_joint]))
        if int(self.pool.nworld) != int(self.nworld):
            raise ValueError(
                f"Dataset/pool mismatch: datasets provide nworld={self.nworld}, but pool.nworld={self.pool.nworld}. "
                "Use one dataset per env, and configure pool nworld accordingly."
            )
        self._replay_cache_key_by_joint = {
            joint: self.pool.prepare_replay_datasets(self._trajs_by_joint[joint], cache_key=f"joint:{joint}")
            for joint in self.joint_order
        }

        # Apply external baseline gains before snapshotting fixed params.
        self._apply_base_gains_to_pool(base_kp=base_kp, base_kv=base_kv)

        self._capture_baseline_parameters()
        self._joint_fixed_params = {joint: self._baseline_joint_param_dict(joint) for joint in self.joint_order}

    def close(self) -> None:
        if self.pool.is_connected:
            self.pool.disconnect()

    def __enter__(self) -> "MJWarpPopulationJointCost":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    def _load_trajectories_by_joint(
        self,
        datasets_by_joint: Mapping[str, Sequence[JointMotionDataset]],
    ) -> dict[str, list[JointMotionDataset]]:
        expected_action_names = [f"{joint}.pos" for joint in JOINT_ORDER]
        out: dict[str, list[JointMotionDataset]] = {}
        for joint in self.joint_order:
            if joint not in datasets_by_joint:
                raise ValueError(f"datasets_by_joint is missing joint '{joint}'")
            seq = list(datasets_by_joint[joint])
            if len(seq) < 1:
                raise ValueError(f"Joint '{joint}' has no datasets.")
            trajs: list[JointMotionDataset] = []
            for traj in seq:
                if not isinstance(traj, JointMotionDataset):
                    raise TypeError(
                        "datasets_by_joint must contain JointMotionDataset objects. "
                        f"Got {type(traj).__name__} for joint '{joint}'."
                    )
                trajs.append(traj)
            for traj in trajs:
                if list(traj.action_names) != expected_action_names:
                    raise ValueError(
                        "Dataset action order mismatch with simulator JOINT_ORDER. "
                        f"dataset_root={traj.dataset_root}"
                    )
            out[joint] = trajs

        counts = {joint: len(v) for joint, v in out.items()}
        n0 = next(iter(counts.values()))
        bad = {joint: n for joint, n in counts.items() if n != n0}
        if bad:
            raise ValueError(f"All joints must have same dataset count. Mismatch: {bad} (reference={n0})")
        return out

    def _capture_baseline_parameters(self) -> None:
        snap = self.pool.get_model_snapshot(0)
        self._base_armature = np.asarray(snap["armature"], dtype=np.float64)
        self._base_viscous = np.asarray(snap["viscous"], dtype=np.float64)
        self._base_dry = np.asarray(snap["dry"], dtype=np.float64)
        self._base_torque_limit = np.asarray(snap["torque_limit"], dtype=np.float64)
        self._base_action_delay_steps = np.asarray(snap["action_delay_steps"], dtype=np.float64)
        self._base_kp = np.asarray(snap["kp"], dtype=np.float64)
        self._base_kv = np.asarray(snap["kv"], dtype=np.float64)

        self._body_name_by_joint = {str(k): str(v) for k, v in snap["body_name_by_joint"].items()}
        self._base_mass_by_joint = {str(k): float(v) for k, v in snap["body_mass_by_joint"].items()}
        self._base_com_by_joint = {
            str(k): np.asarray(v, dtype=np.float64).copy() for k, v in snap["body_com_by_joint"].items()
        }

        for joint in self.joint_order:
            if joint not in self._body_name_by_joint:
                raise RuntimeError(f"Body mapping is missing joint '{joint}' in model snapshot.")
            if joint not in self._base_mass_by_joint:
                raise RuntimeError(f"Body mass baseline is missing joint '{joint}' in model snapshot.")
            if joint not in self._base_com_by_joint:
                raise RuntimeError(f"Body COM baseline is missing joint '{joint}' in model snapshot.")

        body_names = [self._body_name_by_joint[joint] for joint in self.joint_order]
        if len(set(body_names)) != len(body_names):
            raise RuntimeError(
                "Joint-to-body mapping contains duplicates; body mass/COM cannot be independently set per joint."
            )

    def _apply_base_gains_to_pool(self, *, base_kp: Sequence[float], base_kv: Sequence[float]) -> None:
        kp_vec = np.asarray(base_kp, dtype=np.float64).reshape(-1)
        kv_vec = np.asarray(base_kv, dtype=np.float64).reshape(-1)
        if kp_vec.shape != (len(JOINT_ORDER),) or kv_vec.shape != (len(JOINT_ORDER),):
            raise ValueError(
                f"Unexpected base gains shape: kp={kp_vec.shape}, kv={kv_vec.shape}, "
                f"expected {(len(JOINT_ORDER),)}."
            )
        kp_mat = np.repeat(kp_vec[None, :], self.n_model, axis=0)
        kv_mat = np.repeat(kv_vec[None, :], self.n_model, axis=0)
        self.pool.set_joint_gains_per_model(kp=kp_mat, kv=kv_mat, joint_keys=JOINT_ORDER)

    def _baseline_joint_param_dict(self, joint: str) -> dict[str, float]:
        jidx = self._model_joint_index[joint]
        com = self._base_com_by_joint[joint]
        return {
            "armature": float(self._base_armature[jidx]),
            "viscous_friction": float(self._base_viscous[jidx]),
            "dry_friction": float(self._base_dry[jidx]),
            "torque_limit": float(self._base_torque_limit[jidx]),
            "action_delay_steps": float(self._base_action_delay_steps[jidx]),
            "kp": float(self._base_kp[jidx]),
            "kv": float(self._base_kv[jidx]),
            "mass": float(self._base_mass_by_joint[joint]),
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "com_z": float(com[2]),
        }

    def get_joint_fixed_params(self, joint: str) -> dict[str, float]:
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'")
        return dict(self._joint_fixed_params[joint])

    def get_joint_dataset_roots(self, joint: str) -> list[Path]:
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'")
        return [Path(t.dataset_root) for t in self._trajs_by_joint[joint]]

    def get_joint_trajectories(self, joint: str) -> list[JointMotionDataset]:
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'")
        return list(self._trajs_by_joint[joint])

    def apply_candidate_batch_for_joint(
        self,
        *,
        joint: str,
        candidate_params: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'")
        x = self._candidate_matrix(candidate_params)
        self._apply_joint_parameters(active_joint=joint, candidate_mat=x)
        return x

    def replay_joint_datasets(
        self,
        *,
        joint: str,
        reset_before: bool = True,
        initialize_from_first_action: bool = True,
    ) -> dict[str, np.ndarray]:
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'")
        return self.pool.replay_datasets_actions(
            self._trajs_by_joint[joint],
            cache_key=self._replay_cache_key_by_joint[joint],
            reset_before=reset_before,
            initialize_from_first_action=initialize_from_first_action,
        )

    def _to_absolute_joint_param(self, joint: str, pname: str, value: float) -> tuple[str, float]:
        v = float(value)
        if pname == "action_delay_steps":
            return "action_delay_steps", float(int(max(0, round(v))))
        if pname == "mass_scale":
            return "mass", float(self._base_mass_by_joint[joint]) * v
        if pname == "torque_limit_scale":
            jidx = self._model_joint_index[joint]
            return "torque_limit", float(self._base_torque_limit[jidx]) * v
        if pname == "com_x_scale":
            return "com_x", float(self._base_com_by_joint[joint][0]) * v
        if pname == "com_y_scale":
            return "com_y", float(self._base_com_by_joint[joint][1]) * v
        if pname == "com_z_scale":
            return "com_z", float(self._base_com_by_joint[joint][2]) * v
        return pname, v

    def set_joint_fixed_params(self, joint: str, params: Mapping[str, float] | Sequence[float]) -> None:
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'")
        state = self._joint_fixed_params[joint]

        if isinstance(params, Mapping):
            for key, value in params.items():
                ck = _normalize_param_name(key)
                abs_key, abs_val = self._to_absolute_joint_param(joint, ck, float(value))
                state[abs_key] = abs_val
            return

        vec = np.asarray(params, dtype=np.float64).reshape(-1)
        if vec.size != self.layout.nparam:
            raise ValueError(f"Expected {self.layout.nparam} params for joint '{joint}', got {vec.size}")
        for i, pname in enumerate(self.layout.param_order):
            abs_key, abs_val = self._to_absolute_joint_param(joint, pname, float(vec[i]))
            state[abs_key] = abs_val

    def _candidate_matrix(self, candidate_params: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(candidate_params, dtype=np.float64)
        if x.ndim == 1:
            if x.size != self.layout.nparam:
                raise ValueError(f"Expected candidate size {self.layout.nparam}, got {x.size}")
            x = np.repeat(x[None, :], self.n_model, axis=0)
        if x.shape != (self.n_model, self.layout.nparam):
            raise ValueError(f"Expected candidate shape {(self.n_model, self.layout.nparam)}, got {x.shape}")
        return x

    def _apply_joint_parameters(self, active_joint: str, candidate_mat: np.ndarray) -> None:
        n_joint = len(self.joint_order)
        arm = np.zeros((self.n_model, n_joint), dtype=np.float64)
        visc = np.zeros((self.n_model, n_joint), dtype=np.float64)
        dry = np.zeros((self.n_model, n_joint), dtype=np.float64)
        torque = np.zeros((self.n_model, n_joint), dtype=np.float64)
        action_delay_steps = np.zeros((self.n_model, n_joint), dtype=np.float64)
        kp = np.zeros((self.n_model, n_joint), dtype=np.float64)
        kv = np.zeros((self.n_model, n_joint), dtype=np.float64)
        mass = np.zeros((self.n_model, n_joint), dtype=np.float64)
        com = np.zeros((self.n_model, n_joint, 3), dtype=np.float64)

        for jidx, joint in enumerate(self.joint_order):
            s = self._joint_fixed_params[joint]
            arm[:, jidx] = float(s["armature"])
            visc[:, jidx] = float(s["viscous_friction"])
            dry[:, jidx] = float(s["dry_friction"])
            torque[:, jidx] = float(s["torque_limit"])
            action_delay_steps[:, jidx] = float(s["action_delay_steps"])
            kp[:, jidx] = float(s["kp"])
            kv[:, jidx] = float(s["kv"])
            mass[:, jidx] = float(s["mass"])
            com[:, jidx, 0] = float(s["com_x"])
            com[:, jidx, 1] = float(s["com_y"])
            com[:, jidx, 2] = float(s["com_z"])

        active_jidx = self._joint_index[active_joint]

        for pidx, pname in enumerate(self.layout.param_order):
            vals = candidate_mat[:, pidx]
            if pname == "armature":
                arm[:, active_jidx] = vals
            elif pname == "viscous_friction":
                visc[:, active_jidx] = vals
            elif pname == "dry_friction":
                dry[:, active_jidx] = vals
            elif pname == "torque_limit":
                torque[:, active_jidx] = vals
            elif pname == "torque_limit_scale":
                torque[:, active_jidx] = float(self._base_torque_limit[self._model_joint_index[active_joint]]) * vals
            elif pname == "action_delay_steps":
                action_delay_steps[:, active_jidx] = vals
            elif pname == "kp":
                kp[:, active_jidx] = vals
            elif pname == "kv":
                kv[:, active_jidx] = vals
            elif pname == "mass":
                mass[:, active_jidx] = vals
            elif pname == "mass_scale":
                mass[:, active_jidx] = float(self._base_mass_by_joint[active_joint]) * vals
            elif pname == "com_x":
                com[:, active_jidx, 0] = vals
            elif pname == "com_x_scale":
                com[:, active_jidx, 0] = float(self._base_com_by_joint[active_joint][0]) * vals
            elif pname == "com_y":
                com[:, active_jidx, 1] = vals
            elif pname == "com_y_scale":
                com[:, active_jidx, 1] = float(self._base_com_by_joint[active_joint][1]) * vals
            elif pname == "com_z":
                com[:, active_jidx, 2] = vals
            elif pname == "com_z_scale":
                com[:, active_jidx, 2] = float(self._base_com_by_joint[active_joint][2]) * vals
            else:  # pragma: no cover
                raise RuntimeError(f"Unhandled parameter '{pname}'")

        self.pool.set_joint_armature_per_model(armature=arm, joint_keys=self.joint_order)
        self.pool.set_joint_viscous_friction_per_model(viscous=visc, joint_keys=self.joint_order)
        self.pool.set_joint_static_friction_per_model(frictionloss=dry, joint_keys=self.joint_order)
        self.pool.set_joint_torque_limit_per_model(torque_limit=torque, joint_keys=self.joint_order)
        self.pool.set_joint_action_delay_steps_per_model(action_delay_steps=action_delay_steps, joint_keys=self.joint_order)
        self.pool.set_joint_gains_per_model(kp=kp, kv=kv, joint_keys=self.joint_order)

        body_names = [self._body_name_by_joint[joint] for joint in self.joint_order]
        self.pool.set_body_mass_per_model(mass=mass, body_names=body_names)
        self.pool.set_body_com_per_model(com_xyz=com, body_names=body_names)

    def _lsq_per_model_for_joint(
        self,
        *,
        replay: Mapping[str, np.ndarray],
        trajs: Sequence[JointMotionDataset],
        joint: str,
    ) -> np.ndarray:
        pos = np.asarray(replay["joint_pos_deg"], dtype=np.float64)  # (n_model, nworld, T, nj)
        jidx = self._model_joint_index[joint]
        tmax = int(pos.shape[2])
        out = np.zeros((self.n_model,), dtype=np.float64)
        for wid, traj in enumerate(trajs):
            L = int(min(tmax, traj.observed_pos_deg.shape[0]))
            if L <= 0:
                continue
            sim = pos[:, wid, :L, jidx]
            obs = traj.observed_pos_deg[:L, jidx].astype(np.float64, copy=False)[None, :]
            err = sim - obs
            out += np.sum(err * err, axis=1)
        return out

    def cost_for_joint(
        self,
        *,
        joint: str,
        candidate_params: np.ndarray | Sequence[Sequence[float]],
        commit_best: bool = False,
        debug_print: bool = False,
        debug_rows: int = 2,
        log_timing: bool = False,
    ) -> np.ndarray:
        """
        Evaluate one population on one joint.

        candidate_params shape:
          (n_model, n_param), with n_param=len(param_order)

        Returns:
          lsq_per_model shape: (n_model,)
        """
        if joint not in self._joint_index:
            raise ValueError(f"Unknown joint '{joint}'. Expected one of {self.joint_order}")
        if joint not in self._trajs_by_joint:
            raise ValueError(f"No datasets loaded for joint '{joint}'")

        x = self._candidate_matrix(candidate_params)
        if bool(debug_print):
            nshow = int(max(1, min(int(debug_rows), x.shape[0])))
            print(f"[cost:{joint}] candidate_params (first {nshow} rows):")
            print(np.array2string(x[:nshow], precision=6, suppress_small=False))

        if bool(log_timing):
            print(f"[cost:{joint}] apply parameters...", flush=True)
        t0 = time.perf_counter()
        self._apply_joint_parameters(active_joint=joint, candidate_mat=x)
        t_apply = time.perf_counter() - t0
        if bool(debug_print):
            nshow = int(max(1, min(int(debug_rows), self.n_model)))
            p_list = self.pool.get_runtime_joint_params(joint=joint, model_indices=range(nshow))
            print(f"[cost:{joint}] applied params on models (first {nshow}):")
            for mid, p in enumerate(p_list):
                print(
                    f"  model{mid}: arm={p['armature']:.6g} "
                    f"visc={p['viscous']:.6g} dry={p['dry']:.6g} "
                    f"torque={p['torque_limit']:.6g} "
                    f"delay={p['action_delay_steps']:.6g} "
                    f"kp={p['kp']:.6g} kv={p['kv']:.6g}"
                )
            print(f"[cost:{joint}] apply_time={t_apply:.3f}s")

        if bool(log_timing):
            print(f"[cost:{joint}] replay datasets...", flush=True)
        t1 = time.perf_counter()
        replay = self.pool.replay_datasets_actions(
            self._trajs_by_joint[joint],
            cache_key=self._replay_cache_key_by_joint[joint],
            reset_before=True,
            include_action=False,
            include_velocity=False,
        )
        t_replay = time.perf_counter() - t1
        if bool(debug_print):
            tmax = int(np.asarray(replay["sim_time_s"], dtype=np.float64).shape[0])
            print(f"[cost:{joint}] replay_time={t_replay:.3f}s T={tmax}")

        if bool(log_timing):
            print(f"[cost:{joint}] compute LSQ...", flush=True)
        t2 = time.perf_counter()
        cost = self._lsq_per_model_for_joint(replay=replay, trajs=self._trajs_by_joint[joint], joint=joint)
        t_lsq = time.perf_counter() - t2

        if commit_best:
            best_idx = int(np.argmin(cost))
            self.set_joint_fixed_params(joint, x[best_idx])

        if bool(debug_print) or bool(log_timing):
            print(
                f"[cost:{joint}] timings apply={t_apply:.3f}s replay={t_replay:.3f}s "
                f"lsq={t_lsq:.3f}s total={time.perf_counter() - t0:.3f}s",
                flush=True,
            )

        return cost
