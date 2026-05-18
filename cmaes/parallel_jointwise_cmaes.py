from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from identification_2.cmaes.pool_joint_cost import MJWarpPopulationJointCost

BASE_SCALE_ORDER = (
    "armature_scale",
    "viscous_scale",
    "dry_scale",
    "torque_limit_scale",
    "action_delay_steps",
    "mass_scale",
    "com_x_scale",
    "com_y_scale",
    "com_z_scale",
)


@dataclass(kw_only=True)
class JointwiseCMAESConfig:
    popsize: int
    maxiter: int = 100
    sigma0: float = 0.12
    seed: int = 0
    early_stop_std: float = 1.0
    respect_cma_stop: bool = False

    optimize_gain_scale: bool = False
    armature_scale_lb: float = 0.2
    armature_scale_ub: float = 5.0
    viscous_scale_lb: float = 0.2
    viscous_scale_ub: float = 5.0
    dry_scale_lb: float = 0.2
    dry_scale_ub: float = 5.0
    torque_limit_scale_lb: float = 0.2
    torque_limit_scale_ub: float = 5.0
    action_delay_steps_lb: float = 0.0
    action_delay_steps_ub: float = 8.0
    mass_scale_lb: float = 0.9
    mass_scale_ub: float = 1.1
    com_scale_lb: float = 0.9
    com_scale_ub: float = 1.1
    gain_scale_lb: float = 0.5
    gain_scale_ub: float = 2.0

    joint_order: tuple[str, ...] | None = None
    debug_candidates: bool = False
    debug_candidates_rows: int = 3
    progress_every: int = 1


@dataclass(frozen=True)
class JointwiseCMAESResult:
    config: JointwiseCMAESConfig
    results_by_joint: list[dict[str, Any]] = field(default_factory=list)
    final_absolute_params_by_joint: dict[str, dict[str, float]] = field(default_factory=dict)


class ParallelJointwiseCMAES:
    """
    Joint-wise CMA-ES using the pool-backed evaluator for population parallelism.

    One CMA generation evaluates all `popsize` candidates at once through:
      evaluator.cost_for_joint(..., candidate_params=(popsize, nparam))
    """

    def __init__(
        self,
        *,
        evaluator: MJWarpPopulationJointCost,
        config: JointwiseCMAESConfig,
    ):
        self.evaluator = evaluator
        self.config = config

        if int(self.config.popsize) != int(self.evaluator.n_model):
            raise ValueError(
                f"Config/population mismatch: config.popsize={self.config.popsize} "
                f"but evaluator.n_model={self.evaluator.n_model}. "
                "Set popsize equal to the pool n_model."
            )

    def _scale_order(self) -> tuple[str, ...]:
        order = list(BASE_SCALE_ORDER)
        if bool(self.config.optimize_gain_scale):
            order.append("gain_scale")
        return tuple(order)

    def _scale_bounds(self, scale_order: Sequence[str]) -> tuple[list[float], list[float]]:
        del scale_order
        lb = [
            float(self.config.armature_scale_lb),
            float(self.config.viscous_scale_lb),
            float(self.config.dry_scale_lb),
            float(self.config.torque_limit_scale_lb),
            float(self.config.action_delay_steps_lb),
            float(self.config.mass_scale_lb),
            float(self.config.com_scale_lb),
            float(self.config.com_scale_lb),
            float(self.config.com_scale_lb),
        ]
        ub = [
            float(self.config.armature_scale_ub),
            float(self.config.viscous_scale_ub),
            float(self.config.dry_scale_ub),
            float(self.config.torque_limit_scale_ub),
            float(self.config.action_delay_steps_ub),
            float(self.config.mass_scale_ub),
            float(self.config.com_scale_ub),
            float(self.config.com_scale_ub),
            float(self.config.com_scale_ub),
        ]
        if bool(self.config.optimize_gain_scale):
            lb.append(float(self.config.gain_scale_lb))
            ub.append(float(self.config.gain_scale_ub))
        return lb, ub

    def _scales_to_candidate_params(
        self,
        *,
        scales: np.ndarray,
        base_abs: Mapping[str, float],
        scale_order: Sequence[str],
    ) -> np.ndarray:
        x = np.asarray(scales, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(scale_order):
            raise ValueError(f"Expected scales shape (N,{len(scale_order)}), got {x.shape}")

        sidx = {name: i for i, name in enumerate(scale_order)}
        gain_scale = x[:, sidx["gain_scale"]] if "gain_scale" in sidx else np.ones((x.shape[0],), dtype=np.float64)
        arm_s = x[:, sidx["armature_scale"]]
        visc_s = x[:, sidx["viscous_scale"]]
        dry_s = x[:, sidx["dry_scale"]]
        torque_s = x[:, sidx["torque_limit_scale"]]
        delay_steps = x[:, sidx["action_delay_steps"]]
        mass_s = x[:, sidx["mass_scale"]]
        comx_s = x[:, sidx["com_x_scale"]]
        comy_s = x[:, sidx["com_y_scale"]]
        comz_s = x[:, sidx["com_z_scale"]]

        out = np.zeros((x.shape[0], self.evaluator.layout.nparam), dtype=np.float64)
        for i, pname in enumerate(self.evaluator.layout.param_order):
            if pname == "armature":
                out[:, i] = float(base_abs["armature"]) * arm_s
            elif pname == "viscous_friction":
                out[:, i] = float(base_abs["viscous_friction"]) * visc_s
            elif pname == "dry_friction":
                out[:, i] = float(base_abs["dry_friction"]) * dry_s
            elif pname == "torque_limit":
                out[:, i] = float(base_abs["torque_limit"]) * torque_s
            elif pname == "action_delay_steps":
                out[:, i] = delay_steps
            elif pname == "kp":
                out[:, i] = float(base_abs["kp"]) * gain_scale
            elif pname == "kv":
                out[:, i] = float(base_abs["kv"]) * gain_scale
            elif pname == "mass_scale":
                out[:, i] = mass_s
            elif pname == "mass":
                out[:, i] = float(base_abs["mass"]) * mass_s
            elif pname == "com_x_scale":
                out[:, i] = comx_s
            elif pname == "com_y_scale":
                out[:, i] = comy_s
            elif pname == "com_z_scale":
                out[:, i] = comz_s
            elif pname == "com_x":
                out[:, i] = float(base_abs["com_x"]) * comx_s
            elif pname == "com_y":
                out[:, i] = float(base_abs["com_y"]) * comy_s
            elif pname == "com_z":
                out[:, i] = float(base_abs["com_z"]) * comz_s
            else:
                raise ValueError(f"Unsupported layout parameter for scale-based CMAES: '{pname}'")
        return out

    def _run_one_joint(
        self,
        *,
        joint: str,
        joint_seed: int,
        scale_order: Sequence[str],
        bounds_lb: Sequence[float],
        bounds_ub: Sequence[float],
    ) -> dict[str, Any]:
        try:
            import cma
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ParallelJointwiseCMAES requires package `cma`.") from exc

        base_abs = self.evaluator.get_joint_fixed_params(joint)
        x0 = np.ones((len(scale_order),), dtype=np.float64)
        opts = {
            "bounds": [list(bounds_lb), list(bounds_ub)],
            "popsize": int(self.config.popsize),
            "seed": int(joint_seed),
            "verbose": -9,
        }
        es = cma.CMAEvolutionStrategy(x0.tolist(), float(self.config.sigma0), opts)
        print(
            f"[{joint}] start CMA-ES: maxiter={int(self.config.maxiter)} popsize={int(self.config.popsize)} "
            f"sigma0={float(self.config.sigma0):.4g}",
            flush=True,
        )

        history: list[dict[str, float]] = []
        best_cost = float("inf")
        best_scale = np.asarray(x0, dtype=np.float64).copy()
        best_eval = self._scales_to_candidate_params(scales=best_scale[None, :], base_abs=base_abs, scale_order=scale_order)[0]
        std_stop_triggered = False

        for it in range(int(self.config.maxiter)):
            xs = np.asarray(es.ask(), dtype=np.float64)
            if xs.shape[0] != int(self.config.popsize):
                raise RuntimeError(
                    f"CMA ask() returned {xs.shape[0]} candidates but popsize={self.config.popsize}"
                )

            x_eval = self._scales_to_candidate_params(scales=xs, base_abs=base_abs, scale_order=scale_order)
            if bool(self.config.debug_candidates) and it == 0:
                nshow = int(max(1, min(int(self.config.debug_candidates_rows), xs.shape[0])))
                print(f"[{joint}] CMA raw candidates (first {nshow} rows):")
                print(np.array2string(xs[:nshow], precision=6, suppress_small=False))
                print(f"[{joint}] candidates passed to cost() (first {nshow} rows):")
                print(np.array2string(x_eval[:nshow], precision=6, suppress_small=False))

            iter_idx = int(it) + 1
            should_log = (iter_idx == 1) or (iter_idx == int(self.config.maxiter))
            if int(self.config.progress_every) > 0:
                should_log = should_log or (iter_idx % int(self.config.progress_every) == 0)
            t_cost0 = time.perf_counter()
            if should_log:
                print(
                    f"[{joint}] iter {iter_idx}/{int(self.config.maxiter)} evaluating population...",
                    flush=True,
                )
            y = np.asarray(
                self.evaluator.cost_for_joint(
                    joint=joint,
                    candidate_params=x_eval,
                    commit_best=False,
                    debug_print=bool(self.config.debug_candidates) and it == 0,
                    debug_rows=int(self.config.debug_candidates_rows),
                    log_timing=bool(should_log),
                ),
                dtype=np.float64,
            )
            t_cost = time.perf_counter() - t_cost0
            bad = ~np.isfinite(y)
            if np.any(bad):
                y[bad] = 1e12
            es.tell(xs.tolist(), y.tolist())

            gen_best_idx = int(np.argmin(y))
            gen_best = float(y[gen_best_idx])
            gen_mean = float(np.mean(y))
            gen_std = float(np.std(y))
            gen_rng = float(np.max(y) - np.min(y))
            history.append(
                {
                    "iter": float(it),
                    "best": gen_best,
                    "mean": gen_mean,
                    "std": gen_std,
                    "range": gen_rng,
                }
            )

            if gen_best < best_cost:
                best_cost = gen_best
                best_scale = xs[gen_best_idx].copy()
                best_eval = x_eval[gen_best_idx].copy()

            if should_log:
                print(
                    f"[{joint}] iter {iter_idx}/{int(self.config.maxiter)} done "
                    f"best={gen_best:.6g} mean={gen_mean:.6g} std={gen_std:.6g} "
                    f"cost_time={t_cost:.2f}s",
                    flush=True,
                )

            if float(gen_std) < float(self.config.early_stop_std):
                std_stop_triggered = True
                break

            if bool(self.config.respect_cma_stop) and es.stop():
                break

        self.evaluator.set_joint_fixed_params(joint, best_eval)
        best_abs = self.evaluator.get_joint_fixed_params(joint)
        print(
            f"[{joint}] finished best_train_lsq={float(best_cost):.6g}",
            flush=True,
        )

        stop_reason: dict[str, Any]
        try:
            stop_reason = dict(es.stop())
        except Exception:
            stop_reason = {}
        if std_stop_triggered:
            stop_reason["early_stop_std"] = float(self.config.early_stop_std)

        return {
            "joint": joint,
            "best_train_lsq": float(best_cost),
            "best_scale_vector": {k: float(v) for k, v in zip(scale_order, best_scale)},
            "best_eval_vector": {
                k: float(v) for k, v in zip(self.evaluator.layout.param_order, best_eval)
            },
            "best_eval_array": [float(v) for v in best_eval.tolist()],
            "selected_absolute_params": best_abs,
            "datasets": [str(p) for p in self.evaluator.get_joint_dataset_roots(joint)],
            "history": history,
            "stop": stop_reason,
        }

    def run(
        self,
        on_joint_done: Callable[[dict[str, Any], int, int, list[dict[str, Any]]], None] | None = None,
    ) -> JointwiseCMAESResult:
        scale_order = self._scale_order()
        bounds_lb, bounds_ub = self._scale_bounds(scale_order)

        if self.config.joint_order is None:
            joints = list(self.evaluator.joint_order)
        else:
            joints = [str(j) for j in self.config.joint_order]
            bad = [j for j in joints if j not in self.evaluator.joint_order]
            if bad:
                raise ValueError(f"Unknown joints in config.joint_order: {bad}")
        if len(set(joints)) != len(joints):
            raise ValueError("joint_order contains duplicates.")

        results_by_joint: list[dict[str, Any]] = []
        for joint_idx, joint in enumerate(joints):
            print(f"[runner] optimizing joint {joint_idx + 1}/{len(joints)}: {joint}", flush=True)
            out = self._run_one_joint(
                joint=joint,
                joint_seed=int(self.config.seed) + int(joint_idx),
                scale_order=scale_order,
                bounds_lb=bounds_lb,
                bounds_ub=bounds_ub,
            )
            results_by_joint.append(out)
            if on_joint_done is not None:
                on_joint_done(out, int(joint_idx), int(len(joints)), list(results_by_joint))
            print(f"[runner] joint done: {joint}", flush=True)

        final_abs = {joint: self.evaluator.get_joint_fixed_params(joint) for joint in joints}
        return JointwiseCMAESResult(
            config=self.config,
            results_by_joint=results_by_joint,
            final_absolute_params_by_joint=final_abs,
        )
