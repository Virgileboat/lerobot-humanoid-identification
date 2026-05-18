#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from identification_2.cmaes import (
    CANONICAL_PARAM_ORDER,
    JointwiseCMAESConfig,
    MJWarpPopulationJointCost,
    ParallelJointwiseCMAES,
    load_jointwise_experiment_data,
)
from identification_2.models.lerobot_humanoid import ROBOT_NAME
from identification_2.simulator import JOINT_ORDER, HumanoidMJWarpConfig, HumanoidMJWarpModelPool

DEFAULT_OPTIMIZATION_JOINT_ORDER = (
    "right_ankle_roll",
    "right_ankle_pitch",
    "right_knee",
    "right_hipy",
    "right_hipx",
    "right_hipz",
    "left_ankle_roll",
    "left_ankle_pitch",
    "left_knee",
    "left_hipy",
    "left_hipx",
    "left_hipz",
)

# Release baseline dataset bundle.
DEFAULT_DATASETS_ROOT = (
    Path(__file__).resolve().parents[1] / "models" / ROBOT_NAME / "datasets" / "baseline_controller_v1"
)
DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[1] / "results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint-wise pool-parallel CMA-ES for identification_2.")
    p.add_argument("--datasets-root", type=str, default=str(DEFAULT_DATASETS_ROOT))
    p.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment folder names under --datasets-root. Default: auto-detect all.",
    )
    p.add_argument(
        "--dataset-layout",
        type=str,
        choices=["auto", "grouped_by_joint", "per_experiment"],
        default="auto",
        help=(
            "grouped_by_joint: <datasets_root>/<exp>/<joint>/...\n"
            "per_experiment: <datasets_root>/<exp>/meta+data\n"
            "auto: detect one of the above and fail otherwise."
        ),
    )

    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_ROOT / "cmaes_pool_jointwise"))
    p.add_argument("--mjcf-path", type=str, default=str(HumanoidMJWarpConfig().mjcf_path))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--fixed-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sim-dt", type=float, default=0.005)
    p.add_argument("--physics-substeps-per-action", type=int, default=1)
    p.add_argument(
        "--pool-multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one worker process per model.",
    )
    p.add_argument(
        "--pool-mp-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
    )
    p.add_argument(
        "--pool-pin-workers-to-cores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin each multiprocessing worker to a CPU core.",
    )
    p.add_argument(
        "--pool-prefer-physical-cores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When pinning workers, prefer one logical CPU per physical core.",
    )
    p.add_argument(
        "--pool-worker-num-threads",
        type=int,
        default=1,
        help="Host thread limit per worker process (OMP/MKL/OpenBLAS/Torch). 0 disables forcing.",
    )

    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--sigma0", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--early-stop-std", type=float, default=1.0)
    p.add_argument("--respect-cma-stop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--joint-order",
        nargs="+",
        default=list(DEFAULT_OPTIMIZATION_JOINT_ORDER),
        help="Optimization order.",
    )

    p.add_argument("--armature-scale-lb", type=float, default=0.2)
    p.add_argument("--armature-scale-ub", type=float, default=5.0)
    p.add_argument("--viscous-scale-lb", type=float, default=0.2)
    p.add_argument("--viscous-scale-ub", type=float, default=5.0)
    p.add_argument("--dry-scale-lb", type=float, default=0.2)
    p.add_argument("--dry-scale-ub", type=float, default=5.0)
    p.add_argument("--torque-limit-scale-lb", type=float, default=0.2)
    p.add_argument("--torque-limit-scale-ub", type=float, default=5.0)
    p.add_argument("--action-delay-steps-lb", type=float, default=0.0)
    p.add_argument("--action-delay-steps-ub", type=float, default=8.0)
    p.add_argument("--mass-scale-lb", type=float, default=0.9)
    p.add_argument("--mass-scale-ub", type=float, default=1.1)
    p.add_argument("--com-scale-lb", type=float, default=0.9)
    p.add_argument("--com-scale-ub", type=float, default=1.1)
    p.add_argument(
        "--optimize-gain-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize one shared gain scale per active joint (kp/kv).",
    )
    p.add_argument("--gain-scale-lb", type=float, default=0.5)
    p.add_argument("--gain-scale-ub", type=float, default=2.0)

    p.add_argument("--debug-candidates", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug-candidates-rows", type=int, default=3)
    p.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print CMA-ES progress every N iterations per joint (<=0 disables periodic logs).",
    )
    return p.parse_args()


def _validate_joint_action_excitation(
    *,
    datasets_by_joint: dict[str, list[Any]],
    joint_order: list[str],
    eps_deg: float = 1e-6,
    max_other_joint_ratio: float = 0.05,
    max_other_joint_abs_deg: float = 1e-3,
    require_joint_dominance: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Ensure each optimized joint is actually commanded in the associated datasets.
    Returns per-joint amplitude report for traceability.
    """
    report: dict[str, list[dict[str, Any]]] = {}

    def _dataset_label(ds: Any) -> str:
        root = str(getattr(ds, "dataset_root", ""))
        ep = getattr(ds, "episode_index", None)
        if ep is None:
            return root
        return f"{root} [episode_index={int(ep)}]"

    for joint in joint_order:
        jidx = JOINT_ORDER.index(joint)
        rows: list[dict[str, Any]] = []
        for ds in datasets_by_joint[joint]:
            acts = np.asarray(ds.actions_deg, dtype=np.float64)
            if acts.ndim != 2 or acts.shape[1] != len(JOINT_ORDER):
                raise ValueError(
                    f"Invalid action tensor in dataset '{ds.dataset_root}': expected shape (T, {len(JOINT_ORDER)}), "
                    f"got {acts.shape}."
                )
            if acts.shape[0] > 0:
                amp_by_joint = np.max(np.abs(acts), axis=0)
            else:
                amp_by_joint = np.zeros((len(JOINT_ORDER),), dtype=np.float64)
            active_amp = float(amp_by_joint[jidx])
            dominant_idx = int(np.argmax(amp_by_joint)) if amp_by_joint.size > 0 else jidx
            dominant_joint = JOINT_ORDER[dominant_idx]
            dominant_amp = float(amp_by_joint[dominant_idx]) if amp_by_joint.size > 0 else 0.0
            other_amp = amp_by_joint.copy()
            if other_amp.size > 0:
                other_amp[jidx] = 0.0
            max_other_idx = int(np.argmax(other_amp)) if other_amp.size > 0 else jidx
            max_other_joint = JOINT_ORDER[max_other_idx]
            max_other_amp = float(other_amp[max_other_idx]) if other_amp.size > 0 else 0.0
            other_ratio = max_other_amp / max(active_amp, 1e-12)
            rows.append(
                {
                    "dataset_root": str(getattr(ds, "dataset_root", "")),
                    "episode_index": (
                        None if getattr(ds, "episode_index", None) is None else int(getattr(ds, "episode_index"))
                    ),
                    "active_joint_amp_deg": active_amp,
                    "dominant_joint": str(dominant_joint),
                    "dominant_joint_amp_deg": dominant_amp,
                    "max_other_joint": str(max_other_joint),
                    "max_other_joint_amp_deg": max_other_amp,
                    "max_other_over_active_ratio": float(other_ratio),
                }
            )
            if active_amp <= float(eps_deg):
                raise ValueError(
                    f"Joint '{joint}' has zero commanded action in dataset '{_dataset_label(ds)}' "
                    f"(active amp={active_amp:.6g} deg). Dominant commanded joint is '{dominant_joint}' "
                    f"(amp={dominant_amp:.6g} deg). Use datasets where '{joint}' is excited."
                )
            if require_joint_dominance and dominant_joint != joint:
                raise ValueError(
                    f"Joint '{joint}' dataset mismatch in '{_dataset_label(ds)}': dominant commanded joint is "
                    f"'{dominant_joint}' (amp={dominant_amp:.6g} deg), not '{joint}' (amp={active_amp:.6g} deg)."
                )
            if require_joint_dominance and (
                max_other_amp > max(float(max_other_joint_abs_deg), float(max_other_joint_ratio) * active_amp)
            ):
                raise ValueError(
                    f"Joint '{joint}' dataset in '{_dataset_label(ds)}' is not selective enough: "
                    f"max other joint '{max_other_joint}' has amp={max_other_amp:.6g} deg "
                    f"(ratio={other_ratio:.6g} of active)."
                )
        report[joint] = rows
    return report


def _save_joint_artifacts(run_dir: Path, result_by_joint: list[dict[str, Any]]) -> None:
    for out in result_by_joint:
        joint = str(out["joint"])
        joint_dir = run_dir / joint
        joint_dir.mkdir(parents=True, exist_ok=True)
        (joint_dir / "best_joint_result.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
        hist = out.get("history", [])
        if len(hist) > 0:
            lines = ["iter,best,mean,std,range"]
            lines += [
                f"{int(h['iter'])},{h['best']:.10g},{h['mean']:.10g},{h['std']:.10g},{h['range']:.10g}"
                for h in hist
            ]
            (joint_dir / "history.csv").write_text("\n".join(lines) + "\n")


def _save_best_replay_plot(
    *,
    evaluator: MJWarpPopulationJointCost,
    joint: str,
    best_eval: np.ndarray,
    out_png: Path,
    best_train_lsq: float,
) -> None:
    # Apply selected candidate to all models, then replay once for plotting.
    x = np.asarray(best_eval, dtype=np.float64).reshape(1, -1)
    x = np.repeat(x, evaluator.n_model, axis=0)
    evaluator.apply_candidate_batch_for_joint(joint=joint, candidate_params=x)
    replay = evaluator.replay_joint_datasets(joint=joint, reset_before=True)
    trajs = evaluator.get_joint_trajectories(joint)

    sim = np.asarray(replay["joint_pos_deg"], dtype=np.float64)[0]  # model 0, shape (nworld, T, 12)
    act = np.asarray(replay["action_deg"], dtype=np.float64)[0]  # model 0, shape (nworld, T, 12)
    t_sim = np.asarray(replay["sim_time_s"], dtype=np.float64)
    nworld = int(sim.shape[0])
    njoint = len(JOINT_ORDER)
    joint_names = list(JOINT_ORDER)
    model_joint_index = {j: i for i, j in enumerate(JOINT_ORDER)}

    fig, axs = plt.subplots(njoint, nworld, figsize=(5.0 * nworld, 2.3 * njoint), sharex=False)
    if nworld == 1:
        axs = np.expand_dims(axs, axis=1)

    for row_idx, joint_name in enumerate(joint_names):
        jidx = model_joint_index[joint_name]
        for wid in range(nworld):
            ax = axs[row_idx, wid]
            traj = trajs[wid]
            L = int(min(traj.observed_pos_deg.shape[0], sim.shape[1], t_sim.shape[0]))
            if L <= 0:
                continue
            t_obs = np.asarray(traj.timestamps_s[:L], dtype=np.float64)
            y_obs = np.asarray(traj.observed_pos_deg[:L, jidx], dtype=np.float64)
            y_sim = np.asarray(sim[wid, :L, jidx], dtype=np.float64)
            y_act = np.asarray(act[wid, :L, jidx], dtype=np.float64)
            ax.plot(t_obs, y_obs, linewidth=1.8, alpha=0.95, label=f"real {joint_name}")
            ax.plot(t_sim[:L], y_sim, linewidth=1.4, alpha=0.95, label=f"sim {joint_name}")
            ax.plot(t_sim[:L], y_act, linestyle="--", linewidth=1.2, alpha=0.95, label=f"cmd {joint_name}")
            if row_idx == 0:
                ax.set_title(f"env {wid}")
            if wid == 0:
                ax.set_ylabel(f"{joint_name}\n(deg)")
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and wid == (nworld - 1):
                ax.legend(loc="upper right", fontsize=8)

    for wid in range(nworld):
        axs[-1, wid].set_xlabel("time (s)")

    fig.suptitle(
        f"Best replay for optimized '{joint}' (12 x {nworld} grid), train LSQ={best_train_lsq:.6g}",
        y=0.995,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _save_best_replay_target_joint_plot(
    *,
    evaluator: MJWarpPopulationJointCost,
    joint: str,
    best_eval: np.ndarray,
    out_png: Path,
    best_train_lsq: float,
) -> None:
    # Apply selected candidate to all models, then replay once for plotting.
    x = np.asarray(best_eval, dtype=np.float64).reshape(1, -1)
    x = np.repeat(x, evaluator.n_model, axis=0)
    evaluator.apply_candidate_batch_for_joint(joint=joint, candidate_params=x)
    replay = evaluator.replay_joint_datasets(joint=joint, reset_before=True)
    trajs = evaluator.get_joint_trajectories(joint)

    sim = np.asarray(replay["joint_pos_deg"], dtype=np.float64)[0]  # model 0, shape (nworld, T, 12)
    act = np.asarray(replay["action_deg"], dtype=np.float64)[0]  # model 0, shape (nworld, T, 12)
    t_sim = np.asarray(replay["sim_time_s"], dtype=np.float64)
    nworld = int(sim.shape[0])
    jidx = JOINT_ORDER.index(joint)

    fig, axs = plt.subplots(1, nworld, figsize=(5.0 * nworld, 3.2), sharex=False)
    if nworld == 1:
        axs = np.asarray([axs], dtype=object)

    for wid in range(nworld):
        ax = axs[wid]
        traj = trajs[wid]
        L = int(min(traj.observed_pos_deg.shape[0], sim.shape[1], t_sim.shape[0]))
        if L <= 0:
            continue
        t_obs = np.asarray(traj.timestamps_s[:L], dtype=np.float64)
        y_obs = np.asarray(traj.observed_pos_deg[:L, jidx], dtype=np.float64)
        y_sim = np.asarray(sim[wid, :L, jidx], dtype=np.float64)
        y_act = np.asarray(act[wid, :L, jidx], dtype=np.float64)
        ax.plot(t_obs, y_obs, linewidth=1.8, alpha=0.95, label=f"real {joint}")
        ax.plot(t_sim[:L], y_sim, linewidth=1.4, alpha=0.95, label=f"sim {joint}")
        ax.plot(t_sim[:L], y_act, linestyle="--", linewidth=1.2, alpha=0.95, label=f"cmd {joint}")
        ax.set_title(f"env {wid}")
        ax.grid(True, alpha=0.3)
        if wid == 0:
            ax.set_ylabel(f"{joint} (deg)")
        if wid == (nworld - 1):
            ax.legend(loc="upper right", fontsize=8)

    for wid in range(nworld):
        axs[wid].set_xlabel("time (s)")

    fig.suptitle(
        f"Best replay target joint '{joint}' ({nworld} env), train LSQ={best_train_lsq:.6g}",
        y=0.995,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    print("[main] starting run_cmaes_pool_all_joints", flush=True)
    datasets_root = Path(args.datasets_root)
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets-root not found: {datasets_root}")
    print(f"[main] datasets-root: {datasets_root}", flush=True)

    joint_order = [str(j) for j in args.joint_order]
    unknown = [j for j in joint_order if j not in JOINT_ORDER]
    if unknown:
        raise ValueError(f"Unknown joint(s) in --joint-order: {unknown}")
    print(f"[main] optimization joints ({len(joint_order)}): {joint_order}", flush=True)

    print("[main] loading datasets...", flush=True)
    loaded = load_jointwise_experiment_data(
        datasets_root=datasets_root,
        experiment_names=args.experiments,
        joint_order=joint_order,
        dataset_layout=str(args.dataset_layout),
    )
    experiments = loaded.experiment_names
    datasets_by_joint = loaded.datasets_by_joint
    resolved_layout = loaded.dataset_layout
    base_kp = loaded.base_kp
    base_kv = loaded.base_kv
    action_excitation = _validate_joint_action_excitation(
        datasets_by_joint=datasets_by_joint,
        joint_order=joint_order,
        require_joint_dominance=(resolved_layout == "grouped_by_joint" or len(joint_order) == 1),
    )
    nworld = len(datasets_by_joint[joint_order[0]])
    print(
        f"[main] datasets loaded: layout={resolved_layout} experiments={experiments} nworld={nworld}",
        flush=True,
    )

    out_root = Path(args.out_dir)
    run_dir = out_root / f"pool_jointwise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] run_dir: {run_dir}", flush=True)

    sim_cfg = HumanoidMJWarpConfig(
        mjcf_path=Path(args.mjcf_path),
        nworld=int(nworld),
        device=str(args.device),
        fixed_base=bool(args.fixed_base),
        sim_dt=float(args.sim_dt),
        physics_substeps_per_action=int(max(1, args.physics_substeps_per_action)),
    )
    pool = HumanoidMJWarpModelPool(
        sim_cfg,
        n_model=int(args.popsize),
        use_multiprocessing=bool(args.pool_multiprocessing),
        mp_start_method=str(args.pool_mp_start_method),
        pin_workers_to_cores=bool(args.pool_pin_workers_to_cores),
        prefer_physical_cores=bool(args.pool_prefer_physical_cores),
        worker_num_threads=int(args.pool_worker_num_threads),
    )
    print(
        "[main] pool configured: "
        f"n_model={int(args.popsize)} nworld={int(nworld)} device={args.device} "
        f"fixed_base={bool(args.fixed_base)} pool_mp={bool(args.pool_multiprocessing)} "
        f"pin_workers={bool(args.pool_pin_workers_to_cores)} "
        f"worker_threads={int(args.pool_worker_num_threads)}",
        flush=True,
    )

    cma_cfg = JointwiseCMAESConfig(
        popsize=int(args.popsize),
        maxiter=int(args.maxiter),
        sigma0=float(args.sigma0),
        seed=int(args.seed),
        early_stop_std=float(args.early_stop_std),
        respect_cma_stop=bool(args.respect_cma_stop),
        optimize_gain_scale=bool(args.optimize_gain_scale),
        armature_scale_lb=float(args.armature_scale_lb),
        armature_scale_ub=float(args.armature_scale_ub),
        viscous_scale_lb=float(args.viscous_scale_lb),
        viscous_scale_ub=float(args.viscous_scale_ub),
        dry_scale_lb=float(args.dry_scale_lb),
        dry_scale_ub=float(args.dry_scale_ub),
        torque_limit_scale_lb=float(args.torque_limit_scale_lb),
        torque_limit_scale_ub=float(args.torque_limit_scale_ub),
        action_delay_steps_lb=float(args.action_delay_steps_lb),
        action_delay_steps_ub=float(args.action_delay_steps_ub),
        mass_scale_lb=float(args.mass_scale_lb),
        mass_scale_ub=float(args.mass_scale_ub),
        com_scale_lb=float(args.com_scale_lb),
        com_scale_ub=float(args.com_scale_ub),
        gain_scale_lb=float(args.gain_scale_lb),
        gain_scale_ub=float(args.gain_scale_ub),
        joint_order=tuple(joint_order),
        debug_candidates=bool(args.debug_candidates),
        debug_candidates_rows=int(args.debug_candidates_rows),
        progress_every=int(args.progress_every),
    )
    print(
        "[main] cma config: "
        f"popsize={cma_cfg.popsize} maxiter={cma_cfg.maxiter} sigma0={cma_cfg.sigma0} "
        f"progress_every={cma_cfg.progress_every}",
        flush=True,
    )

    cfg_dump = {
        "datasets_root": str(datasets_root),
        "dataset_layout": resolved_layout,
        "experiments": experiments,
        "joint_order": joint_order,
        "simulator": {
            "mjcf_path": str(sim_cfg.mjcf_path),
            "device": str(sim_cfg.device),
            "fixed_base": bool(sim_cfg.fixed_base),
            "sim_dt": float(sim_cfg.sim_dt),
            "physics_substeps_per_action": int(sim_cfg.physics_substeps_per_action),
            "pool_multiprocessing": bool(args.pool_multiprocessing),
            "pool_mp_start_method": str(args.pool_mp_start_method),
            "pool_pin_workers_to_cores": bool(args.pool_pin_workers_to_cores),
            "pool_prefer_physical_cores": bool(args.pool_prefer_physical_cores),
            "pool_worker_num_threads": int(args.pool_worker_num_threads),
            "n_model": int(args.popsize),
            "nworld": int(nworld),
        },
        "cmaes": asdict(cma_cfg),
        "param_order": list(CANONICAL_PARAM_ORDER),
        "base_gains": {
            "joint_order": list(JOINT_ORDER),
            "kp": [float(v) for v in base_kp.tolist()],
            "kv": [float(v) for v in base_kv.tolist()],
        },
        "action_excitation": action_excitation,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dump, sort_keys=False))
    print("[main] wrote config.yaml", flush=True)

    print("[main] creating evaluator...", flush=True)
    with MJWarpPopulationJointCost(
        pool=pool,
        datasets_by_joint=datasets_by_joint,
        base_kp=base_kp,
        base_kv=base_kv,
        joint_order=joint_order,
        param_order=CANONICAL_PARAM_ORDER,
    ) as evaluator:
        if bool(args.pool_multiprocessing):
            print(f"[main] worker cpu affinity ids: {pool.worker_cpu_ids}", flush=True)
        print("[main] evaluator ready, starting optimization...", flush=True)
        runner = ParallelJointwiseCMAES(evaluator=evaluator, config=cma_cfg)

        def _save_after_joint(
            out: dict[str, Any],
            joint_idx: int,
            n_joint: int,
            results_so_far: list[dict[str, Any]],
        ) -> None:
            del joint_idx, n_joint
            joint = str(out["joint"])
            plot_path = run_dir / joint / "best_replay_12xN.png"
            target_plot_path = run_dir / joint / "best_replay_target_joint.png"
            _save_best_replay_plot(
                evaluator=evaluator,
                joint=joint,
                best_eval=np.asarray(out["best_eval_array"], dtype=np.float64),
                out_png=plot_path,
                best_train_lsq=float(out["best_train_lsq"]),
            )
            _save_best_replay_target_joint_plot(
                evaluator=evaluator,
                joint=joint,
                best_eval=np.asarray(out["best_eval_array"], dtype=np.float64),
                out_png=target_plot_path,
                best_train_lsq=float(out["best_train_lsq"]),
            )
            out["best_replay_plot"] = str(plot_path)
            out["best_replay_target_joint_plot"] = str(target_plot_path)
            _save_joint_artifacts(run_dir, [out])
            final_abs_so_far = {
                str(r["joint"]): evaluator.get_joint_fixed_params(str(r["joint"]))
                for r in results_so_far
            }
            partial_summary = {
                "config": cfg_dump,
                "results_by_joint": results_so_far,
                "final_absolute_params_by_joint": final_abs_so_far,
            }
            partial_path = run_dir / "summary_partial.yaml"
            partial_path.write_text(yaml.safe_dump(partial_summary, sort_keys=False))
            print(f"[main] saved joint artifacts + plots for {joint}", flush=True)

        result = runner.run(on_joint_done=_save_after_joint)
        print("[main] optimization complete", flush=True)

    print("[main] all per-joint artifacts were saved incrementally", flush=True)

    summary = {
        "config": cfg_dump,
        "results_by_joint": result.results_by_joint,
        "final_absolute_params_by_joint": result.final_absolute_params_by_joint,
    }
    summary_path = run_dir / "summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f"Done. Outputs: {run_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
