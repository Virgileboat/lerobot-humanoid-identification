from __future__ import annotations

from typing import Mapping

ROBOT_NAME = "lerobot_humanoid"

# Canonical humanoid joint order used by identification_2.
JOINT_ORDER = (
    "left_hipz",
    "left_hipx",
    "left_hipy",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hipz",
    "right_hipx",
    "right_hipy",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
)
_EXPECTED_MOTOR_IDS = tuple(range(1, len(JOINT_ORDER) + 1))

# Baseline gains used for real-robot data acquisition (per motor id 1..12).
POSITION_KP_BY_MOTOR_ID_BASELINE: dict[int, float] = {
    1: 10.0,
    2: 20.0,
    3: 2.0,
    4: 2.0,
    5: 10.0,
    6: 10.0,
    7: 10.0,
    8: 20.0,
    9: 2.0,
    10: 2.0,
    11: 10.0,
    12: 10.0,
}
POSITION_KD_BY_MOTOR_ID_BASELINE: dict[int, float] = {
    1: 2.0,
    2: 2.0,
    3: 0.1,
    4: 0.1,
    5: 0.5,
    6: 0.5,
    7: 2.0,
    8: 2.0,
    9: 0.1,
    10: 0.1,
    11: 0.5,
    12: 0.5,
}


def load_position_gains_from_baseline(
    *,
    kp_by_motor_id: Mapping[int, float] = POSITION_KP_BY_MOTOR_ID_BASELINE,
    kd_by_motor_id: Mapping[int, float] = POSITION_KD_BY_MOTOR_ID_BASELINE,
) -> tuple[dict[int, float], dict[int, float]]:
    kp = {int(k): float(v) for k, v in kp_by_motor_id.items()}
    kd = {int(k): float(v) for k, v in kd_by_motor_id.items()}
    if tuple(sorted(kp.keys())) != _EXPECTED_MOTOR_IDS:
        raise RuntimeError(f"Invalid baseline kp motor ids: got {tuple(sorted(kp.keys()))}, expected {_EXPECTED_MOTOR_IDS}")
    if tuple(sorted(kd.keys())) != _EXPECTED_MOTOR_IDS:
        raise RuntimeError(f"Invalid baseline kd motor ids: got {tuple(sorted(kd.keys()))}, expected {_EXPECTED_MOTOR_IDS}")
    return kp, kd

# Backward-compatible names used elsewhere in identification_2.
POSITION_KP_BY_MOTOR_ID, POSITION_KD_BY_MOTOR_ID = load_position_gains_from_baseline()
