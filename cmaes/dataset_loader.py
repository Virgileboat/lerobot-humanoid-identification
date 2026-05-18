from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from identification_2.simulator.config import JOINT_ORDER


@dataclass(frozen=True)
class JointMotionDataset:
    dataset_root: Path
    fps: float
    action_names: list[str]
    observation_state_names: list[str]
    actions_deg: np.ndarray  # [T, Nj]
    observed_pos_deg: np.ndarray  # [T, Nj]
    observed_vel_deg_s: np.ndarray  # [T, Nj]
    timestamps_s: np.ndarray  # [T]
    episode_index: int | None = None
    inferred_joint: str | None = None


@dataclass(frozen=True)
class JointwiseExperimentData:
    experiment_names: list[str]
    dataset_layout: str
    datasets_by_joint: dict[str, list[JointMotionDataset]]
    base_kp: np.ndarray
    base_kv: np.ndarray


@dataclass(frozen=True)
class ExperimentJointDatasets:
    experiment_root: Path
    base_kp: np.ndarray
    base_kv: np.ndarray
    datasets_by_joint: dict[str, list[JointMotionDataset]]


def _require_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "load_joint_motion_dataset requires pandas with parquet support."
        ) from exc
    return pd


def load_joint_motion_dataset(dataset_root: str | Path) -> JointMotionDataset:
    """
    Load one local LeRobot dataset episode from:
      <root>/data/chunk-000/file-000.parquet
    """
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    parquet_path = root / "data" / "chunk-000" / "file-000.parquet"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing dataset data file: {parquet_path}")

    info = json.loads(info_path.read_text())
    action_names = list(info["features"]["action"]["names"])
    obs_state_names = list(info["features"]["observation.state"]["names"])
    fps = float(info["fps"])

    pd = _require_pandas()
    df = pd.read_parquet(parquet_path)

    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    obs_state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float32)

    pos_idx: list[int] = []
    vel_idx: list[int] = []
    for joint_name in action_names:
        if not joint_name.endswith(".pos"):
            raise ValueError(f"Unexpected action name format '{joint_name}', expected '<joint>.pos'")
        joint = joint_name[: -len(".pos")]
        pos_key = f"{joint}.pos"
        vel_key = f"{joint}.vel"
        pos_idx.append(obs_state_names.index(pos_key))
        vel_idx.append(obs_state_names.index(vel_key))

    observed_pos_deg = obs_state[:, pos_idx]
    observed_vel_deg_s = obs_state[:, vel_idx]

    return JointMotionDataset(
        dataset_root=root,
        fps=fps,
        action_names=action_names,
        observation_state_names=obs_state_names,
        actions_deg=actions,
        observed_pos_deg=observed_pos_deg,
        observed_vel_deg_s=observed_vel_deg_s,
        timestamps_s=timestamps,
        episode_index=None,
        inferred_joint=None,
    )


def _normalize_task_labels(raw_tasks: Any) -> list[str]:
    if raw_tasks is None:
        return []
    if isinstance(raw_tasks, str):
        return [raw_tasks]
    if isinstance(raw_tasks, (list, tuple, set)):
        return [str(x) for x in raw_tasks]
    arr = np.asarray(raw_tasks, dtype=object).reshape(-1)
    return [str(x) for x in arr.tolist()]


def _infer_joint_from_tasks(task_labels: Sequence[str]) -> str | None:
    for label in task_labels:
        s = str(label)
        for joint in JOINT_ORDER:
            if s.endswith(f":{joint}") or s.endswith(f"/{joint}") or s == joint or joint in s:
                return joint
    return None


def _infer_joint_from_actions(actions_deg: np.ndarray, eps_deg: float = 1e-6) -> str | None:
    acts = np.asarray(actions_deg, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] != len(JOINT_ORDER):
        return None
    if acts.shape[0] < 1:
        return None
    amp = np.max(np.abs(acts), axis=0)
    idx = int(np.argmax(amp))
    if float(amp[idx]) <= float(eps_deg):
        return None
    return JOINT_ORDER[idx]


def load_joint_motion_datasets_in_experiment(experiment_root: str | Path) -> list[JointMotionDataset]:
    """
    Load all internal datasets (episodes) from one experiment folder:
      <exp_root>/data/chunk-000/file-000.parquet

    Returns one JointMotionDataset per episode_index.
    """
    root = Path(experiment_root)
    info_path = root / "meta" / "info.json"
    parquet_path = root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing dataset data file: {parquet_path}")

    info = json.loads(info_path.read_text())
    action_names = list(info["features"]["action"]["names"])
    obs_state_names = list(info["features"]["observation.state"]["names"])
    fps = float(info["fps"])

    pd = _require_pandas()
    df = pd.read_parquet(parquet_path)
    if "episode_index" not in df.columns:
        raise ValueError(f"Dataset '{root}' is missing 'episode_index' column.")

    task_labels_by_episode: dict[int, list[str]] = {}
    if episodes_path.exists():
        ep_df = pd.read_parquet(episodes_path)
        if "episode_index" in ep_df.columns and "tasks" in ep_df.columns:
            for _, row in ep_df.iterrows():
                eid = int(row["episode_index"])
                task_labels_by_episode[eid] = _normalize_task_labels(row["tasks"])

    pos_idx: list[int] = []
    vel_idx: list[int] = []
    for joint_name in action_names:
        if not joint_name.endswith(".pos"):
            raise ValueError(f"Unexpected action name format '{joint_name}', expected '<joint>.pos'")
        joint = joint_name[: -len(".pos")]
        pos_key = f"{joint}.pos"
        vel_key = f"{joint}.vel"
        pos_idx.append(obs_state_names.index(pos_key))
        vel_idx.append(obs_state_names.index(vel_key))

    out: list[JointMotionDataset] = []
    episode_ids = sorted(int(x) for x in df["episode_index"].unique().tolist())
    for eid in episode_ids:
        sub = df[df["episode_index"] == eid]
        if "index" in sub.columns:
            sub = sub.sort_values(by="index")
        elif "frame_index" in sub.columns:
            sub = sub.sort_values(by="frame_index")
        else:
            sub = sub.sort_values(by="timestamp")

        actions = np.stack(sub["action"].to_numpy()).astype(np.float32)
        obs_state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
        timestamps = sub["timestamp"].to_numpy(dtype=np.float32)
        observed_pos_deg = obs_state[:, pos_idx]
        observed_vel_deg_s = obs_state[:, vel_idx]

        task_labels = task_labels_by_episode.get(eid, [])
        inferred_joint = _infer_joint_from_tasks(task_labels) or _infer_joint_from_actions(actions)
        out.append(
            JointMotionDataset(
                dataset_root=root,
                fps=fps,
                action_names=action_names,
                observation_state_names=obs_state_names,
                actions_deg=actions,
                observed_pos_deg=observed_pos_deg,
                observed_vel_deg_s=observed_vel_deg_s,
                timestamps_s=timestamps,
                episode_index=eid,
                inferred_joint=inferred_joint,
            )
        )
    return out


def _is_dataset_root(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data" / "chunk-000").exists()


def _is_grouped_joint_layout(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / joint).is_dir() and _is_dataset_root(path / joint) for joint in JOINT_ORDER)


def resolve_experiment_names(
    datasets_root: str | Path,
    requested: Sequence[str] | None = None,
) -> list[str]:
    root = Path(datasets_root)
    if not root.exists():
        raise FileNotFoundError(f"Datasets root not found: {root}")

    if requested is not None and len(requested) > 0:
        out = [str(x) for x in requested]
        for exp in out:
            p = root / exp
            if not p.exists():
                raise FileNotFoundError(f"Experiment folder not found: {p}")
        return out

    names: list[str] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        per_experiment = _is_dataset_root(p)
        grouped_by_joint = any((c.is_dir() and _is_dataset_root(c)) for c in p.iterdir())
        if per_experiment or grouped_by_joint:
            names.append(p.name)
    if len(names) < 1:
        raise RuntimeError(
            f"No experiments found under {root}. "
            "Expected either grouped_by_joint (<exp>/<joint>/meta+data) "
            "or per_experiment (<exp>/meta+data)."
        )
    return names


def load_experiment_base_gains(
    experiment_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    exp_root = Path(experiment_root)
    payload: dict[str, Any] | None = None

    # New format: <experiment_root>/acquisition_context.json
    acq_root = exp_root / "acquisition_context.json"
    if acq_root.exists():
        payload = json.loads(acq_root.read_text())

    # Legacy format: <experiment_root>/meta/acquisition_context.json or info.json["identification_2"]
    if payload is None:
        meta_dir = exp_root / "meta"
        acq_path = meta_dir / "acquisition_context.json"
        info_path = meta_dir / "info.json"
        if acq_path.exists():
            payload = json.loads(acq_path.read_text())
        elif info_path.exists():
            info = json.loads(info_path.read_text())
            maybe = info.get("identification_2")
            if isinstance(maybe, dict):
                payload = maybe

    # Legacy grouped-by-joint fallback: any child joint dataset may contain metadata.
    if payload is None and exp_root.is_dir():
        for child in sorted(exp_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or not _is_dataset_root(child):
                continue
            child_acq = child / "meta" / "acquisition_context.json"
            child_info = child / "meta" / "info.json"
            if child_acq.exists():
                payload = json.loads(child_acq.read_text())
                break
            if child_info.exists():
                info = json.loads(child_info.read_text())
                maybe = info.get("identification_2")
                if isinstance(maybe, dict):
                    payload = maybe
                    break
    if payload is None:
        raise ValueError(
            f"Missing gains metadata for experiment '{exp_root}'. "
            "Expected acquisition_context.json at experiment root "
            "or legacy meta/acquisition_context.json / meta/info.json['identification_2']."
        )

    kp_raw = payload.get("position_kp_by_motor_id")
    kd_raw = payload.get("position_kd_by_motor_id")
    if not isinstance(kp_raw, dict) or not isinstance(kd_raw, dict):
        raise ValueError(
            f"Invalid gains metadata in experiment '{exp_root}': "
            "expected position_kp_by_motor_id and position_kd_by_motor_id mappings."
        )

    expected_ids = tuple(range(1, len(JOINT_ORDER) + 1))
    kp_by_motor = {int(k): float(v) for k, v in kp_raw.items()}
    kv_by_motor = {int(k): float(v) for k, v in kd_raw.items()}
    if tuple(sorted(kp_by_motor.keys())) != expected_ids:
        raise ValueError(
            f"Incomplete kp motor IDs in experiment '{exp_root}'. "
            f"Expected IDs {expected_ids}, got {tuple(sorted(kp_by_motor.keys()))}."
        )
    if tuple(sorted(kv_by_motor.keys())) != expected_ids:
        raise ValueError(
            f"Incomplete kd/kv motor IDs in experiment '{exp_root}'. "
            f"Expected IDs {expected_ids}, got {tuple(sorted(kv_by_motor.keys()))}."
        )

    kp = np.asarray([kp_by_motor[mid] for mid in expected_ids], dtype=np.float64)
    kv = np.asarray([kv_by_motor[mid] for mid in expected_ids], dtype=np.float64)
    return kp, kv


def load_experiment_joint_datasets(
    experiment_root: str | Path,
    *,
    joint_order: Sequence[str] | None = None,
) -> ExperimentJointDatasets:
    """
    Load one experiment folder and return:
      - base gains used in this experiment
      - all datasets produced in this experiment, grouped by joint name
    """
    exp_root = Path(experiment_root)
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_root}")
    if not exp_root.is_dir():
        raise ValueError(f"Experiment path is not a directory: {exp_root}")

    joints = [str(j) for j in (list(joint_order) if joint_order is not None else list(JOINT_ORDER))]
    if len(joints) < 1:
        raise ValueError("joint_order must contain at least one joint.")
    unknown = [j for j in joints if j not in JOINT_ORDER]
    if unknown:
        raise ValueError(f"Unknown joint(s): {unknown}")

    base_kp, base_kv = load_experiment_base_gains(exp_root)
    out: dict[str, list[JointMotionDataset]] = {joint: [] for joint in joints}

    if _is_grouped_joint_layout(exp_root):
        for joint in joints:
            p = exp_root / joint
            if _is_dataset_root(p):
                out[joint].append(load_joint_motion_dataset(p))
    elif _is_dataset_root(exp_root):
        datasets_in_exp = load_joint_motion_datasets_in_experiment(exp_root)
        for ds in datasets_in_exp:
            inferred_joint = ds.inferred_joint or _infer_joint_from_actions(ds.actions_deg)
            if inferred_joint in out:
                out[inferred_joint].append(ds)
    else:
        raise RuntimeError(
            f"Unsupported experiment layout: {exp_root}. "
            "Expected grouped_by_joint (<exp>/<joint>/meta+data) "
            "or per_experiment (<exp>/meta+data with episode_index)."
        )

    # Keep only joints that have produced datasets.
    compact = {joint: seq for joint, seq in out.items() if len(seq) > 0}
    return ExperimentJointDatasets(
        experiment_root=exp_root,
        base_kp=base_kp,
        base_kv=base_kv,
        datasets_by_joint=compact,
    )


def combine_experiment_joint_datasets(
    experiments_data: Sequence[ExperimentJointDatasets],
    *,
    joint_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[JointMotionDataset]]]:
    """
    Combine several experiments for cost evaluation.
    Constraints:
      - all experiments must use the same base gains
      - for each requested joint, each experiment must provide exactly one dataset
    """
    items = list(experiments_data)
    if len(items) < 1:
        raise ValueError("No experiments provided.")

    joints = [str(j) for j in joint_order]
    if len(joints) < 1:
        raise ValueError("joint_order must contain at least one joint.")

    ref = items[0]
    ref_kp = np.asarray(ref.base_kp, dtype=np.float64)
    ref_kv = np.asarray(ref.base_kv, dtype=np.float64)
    for exp in items[1:]:
        same_kp = np.allclose(np.asarray(exp.base_kp, dtype=np.float64), ref_kp, rtol=0.0, atol=1e-9)
        same_kv = np.allclose(np.asarray(exp.base_kv, dtype=np.float64), ref_kv, rtol=0.0, atol=1e-9)
        if not (same_kp and same_kv):
            raise ValueError(
                "Experiment gains mismatch across selected experiments. "
                f"Reference: {ref.experiment_root}, mismatch: {exp.experiment_root}."
            )

    by_joint: dict[str, list[JointMotionDataset]] = {joint: [] for joint in joints}
    for joint in joints:
        for exp in items:
            seq = list(exp.datasets_by_joint.get(joint, []))
            if len(seq) < 1:
                available = sorted(exp.datasets_by_joint.keys())
                raise RuntimeError(
                    f"Experiment '{exp.experiment_root}' has no dataset associated to joint '{joint}'. "
                    f"Available joints: {available}."
                )
            if len(seq) > 1:
                raise RuntimeError(
                    f"Experiment '{exp.experiment_root}' has multiple datasets associated to joint '{joint}' "
                    f"({len(seq)} found). Expected exactly one dataset per joint per experiment."
                )
            by_joint[joint].append(seq[0])
    return ref_kp, ref_kv, by_joint


def load_jointwise_experiment_data(
    *,
    datasets_root: str | Path,
    experiment_names: Sequence[str] | None = None,
    joint_order: Sequence[str] | None = None,
    dataset_layout: str = "auto",
) -> JointwiseExperimentData:
    root = Path(datasets_root)
    joints = [str(j) for j in (list(joint_order) if joint_order is not None else list(JOINT_ORDER))]
    exps = resolve_experiment_names(root, requested=experiment_names)
    per_exp: list[ExperimentJointDatasets] = []
    layout_seen: set[str] = set()
    for exp in exps:
        exp_root = root / exp
        grouped = _is_grouped_joint_layout(exp_root)
        per_exp_layout = "grouped_by_joint" if grouped else "per_experiment"
        if dataset_layout == "grouped_by_joint" and per_exp_layout != "grouped_by_joint":
            raise RuntimeError(f"Experiment '{exp_root}' is not grouped_by_joint.")
        if dataset_layout == "per_experiment" and per_exp_layout != "per_experiment":
            raise RuntimeError(f"Experiment '{exp_root}' is not per_experiment.")
        layout_seen.add(per_exp_layout)
        per_exp.append(load_experiment_joint_datasets(exp_root, joint_order=joints))
    if len(layout_seen) > 1:
        raise RuntimeError(
            f"Mixed dataset layouts are not supported in one run: {sorted(layout_seen)}."
        )
    resolved_layout = next(iter(layout_seen)) if len(layout_seen) == 1 else str(dataset_layout)
    base_kp, base_kv, datasets_by_joint = combine_experiment_joint_datasets(
        per_exp,
        joint_order=joints,
    )
    return JointwiseExperimentData(
        experiment_names=exps,
        dataset_layout=resolved_layout,
        datasets_by_joint=datasets_by_joint,
        base_kp=base_kp,
        base_kv=base_kv,
    )
