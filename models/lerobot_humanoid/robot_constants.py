from __future__ import annotations

"""
Compatibility shim.

Canonical robot constants now live in `identification_2.simulator.robot_spec`.
Keep this module to avoid breaking existing imports.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

from identification_2.simulator.mjcf_paths import get_default_mjcf_path
from identification_2.simulator.robot_spec import (
    JOINT_ORDER,
    POSITION_KD_BY_MOTOR_ID,
    POSITION_KD_BY_MOTOR_ID_BASELINE,
    POSITION_KP_BY_MOTOR_ID,
    POSITION_KP_BY_MOTOR_ID_BASELINE,
    ROBOT_NAME,
    load_position_gains_from_baseline,
)

# Motor IDs per leg:
# hipz=1/7, hipx=2/8, hipy=3/9, knee=4/10, ankle motors=(5,6)/(11,12)
_MOTOR_ID_BY_ACTUATOR_NAME: dict[str, int] = {
    "m_hipz_left": 1,
    "m_hipx_left": 2,
    "m_hipy_left": 3,
    "m_knee_left": 4,
    "m_ankley_left": 5,
    "m_anklex_left": 6,
    "m_hipz_right": 7,
    "m_hipx_right": 8,
    "m_hipy_right": 9,
    "m_knee_right": 10,
    "m_ankley_right": 11,
    "m_anklex_right": 12,
}
_EXPECTED_MOTOR_IDS = tuple(sorted(_MOTOR_ID_BY_ACTUATOR_NAME.values()))
DEFAULT_GAINS_MJCF_PATH = get_default_mjcf_path(fixed_base=False)


def load_position_gains_from_model_spec(
    mjcf_path: str | Path = DEFAULT_GAINS_MJCF_PATH,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Read actuator gains from the robot model specification (MJCF actuator block).

    `kp` is read from MJCF `kp` and `kd` is mapped from MJCF `kv`.
    """
    path = Path(mjcf_path)
    if not path.exists():
        raise FileNotFoundError(f"Gains MJCF file not found: {path}")

    root = ET.parse(path).getroot()
    kp_by_motor_id: dict[int, float] = {}
    kd_by_motor_id: dict[int, float] = {}
    for elem in root.findall(".//actuator/position"):
        actuator_name = str(elem.attrib.get("name", ""))
        motor_id = _MOTOR_ID_BY_ACTUATOR_NAME.get(actuator_name)
        if motor_id is None:
            continue
        if "kp" not in elem.attrib:
            raise RuntimeError(f"Missing 'kp' for actuator '{actuator_name}' in {path}")
        if "kv" not in elem.attrib:
            raise RuntimeError(f"Missing 'kv' for actuator '{actuator_name}' in {path}")
        kp_by_motor_id[motor_id] = float(elem.attrib["kp"])
        kd_by_motor_id[motor_id] = float(elem.attrib["kv"])

    found_ids = tuple(sorted(kp_by_motor_id.keys()))
    if found_ids != _EXPECTED_MOTOR_IDS:
        missing = [mid for mid in _EXPECTED_MOTOR_IDS if mid not in kp_by_motor_id]
        raise RuntimeError(
            f"Incomplete gains mapping from {path}. Missing motor IDs: {missing}. Found: {found_ids}"
        )

    return kp_by_motor_id, kd_by_motor_id


__all__ = [
    "ROBOT_NAME",
    "DEFAULT_GAINS_MJCF_PATH",
    "JOINT_ORDER",
    "POSITION_KP_BY_MOTOR_ID_BASELINE",
    "POSITION_KD_BY_MOTOR_ID_BASELINE",
    "POSITION_KP_BY_MOTOR_ID",
    "POSITION_KD_BY_MOTOR_ID",
    "load_position_gains_from_baseline",
    "load_position_gains_from_model_spec",
]
