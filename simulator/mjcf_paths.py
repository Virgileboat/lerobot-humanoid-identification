from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


MODEL_NAME = "bipedal_plateform_no_arms"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _submodule_src_dir() -> Path:
    return _repo_root() / "lerobot-humanoid-models" / "src"


def _submodule_mjcf_dir() -> Path:
    return _repo_root() / "lerobot-humanoid-models" / "models" / MODEL_NAME / "mjcf"


def _add_submodule_src_to_pythonpath() -> None:
    src = _submodule_src_dir()
    if src.is_dir():
        src_s = str(src)
        if src_s not in sys.path:
            sys.path.insert(0, src_s)


def _external_mjcf_dir() -> Path | None:
    _add_submodule_src_to_pythonpath()
    try:
        from lerobot_humanoid_models.bipedal_plateform_no_arms.constants import MJCF_DIR
    except Exception:
        out = _submodule_mjcf_dir()
        return out if out.is_dir() else None
    out = Path(MJCF_DIR)
    if out.is_dir():
        return out
    fallback = _submodule_mjcf_dir()
    if fallback.is_dir():
        return fallback
    return None


def get_model_mjcf_dir(*, prefer_external: bool = True) -> Path:
    if prefer_external:
        ext = _external_mjcf_dir()
        if ext is not None:
            return ext
    raise FileNotFoundError(
        "Could not resolve MJCF directory from the submodule dependency "
        f"({_submodule_mjcf_dir()})."
    )


def _rewrite_robot_to_fixed_base(src_robot_xml: Path, dst_robot_fixed_xml: Path) -> int:
    tree = ET.parse(src_robot_xml)
    root = tree.getroot()
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "freejoint":
                parent.remove(child)
                removed += 1
    if removed < 1:
        raise RuntimeError(f"No <freejoint> found in robot model: {src_robot_xml}")
    dst_robot_fixed_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_robot_fixed_xml, encoding="utf-8", xml_declaration=True)
    return removed


def _rewrite_scene_to_fixed_base(src_scene_xml: Path, dst_scene_fixed_xml: Path) -> None:
    tree = ET.parse(src_scene_xml)
    root = tree.getroot()
    replaced = 0
    for elem in root.iter("include"):
        if elem.attrib.get("file") == "robot.xml":
            elem.set("file", "robot_fixed_base.xml")
            replaced += 1
    if replaced < 1:
        raise RuntimeError(
            f"Could not build fixed-base scene from {src_scene_xml}: expected include file=\"robot.xml\"."
        )
    dst_scene_fixed_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_scene_fixed_xml, encoding="utf-8", xml_declaration=True)


def _generated_fixed_base_dir(scene_xml: Path) -> Path:
    digest = hashlib.sha1(str(scene_xml.resolve()).encode("utf-8")).hexdigest()[:10]
    return _repo_root() / ".cache" / "generated_mjcf" / f"{scene_xml.stem}_{digest}"


def ensure_fixed_base_scene(scene_xml: Path) -> Path:
    src_scene = Path(scene_xml).resolve()
    if not src_scene.is_file():
        raise FileNotFoundError(f"Scene XML not found: {src_scene}")

    out_dir = _generated_fixed_base_dir(src_scene)
    out_robot_fixed = out_dir / "robot_fixed_base.xml"
    out_scene_fixed = out_dir / "scene_fixed_base.xml"

    src_robot = src_scene.parent / "robot.xml"
    if not src_robot.is_file():
        raise FileNotFoundError(f"robot.xml not found next to scene.xml: {src_robot}")

    # Regenerate when missing or source changed.
    need_robot_regen = (not out_robot_fixed.exists()) or (
        out_robot_fixed.stat().st_mtime < src_robot.stat().st_mtime
    )
    if need_robot_regen:
        _rewrite_robot_to_fixed_base(src_robot, out_robot_fixed)
    need_scene_regen = (not out_scene_fixed.exists()) or (
        out_scene_fixed.stat().st_mtime < src_scene.stat().st_mtime
    ) or need_robot_regen
    if need_scene_regen:
        _rewrite_scene_to_fixed_base(src_scene, out_scene_fixed)

    return out_scene_fixed


def get_default_mjcf_path(*, fixed_base: bool, prefer_external: bool = True) -> Path:
    mjcf_dir = get_model_mjcf_dir(prefer_external=prefer_external)

    if fixed_base:
        for candidate in (
            "sim_scene_safe_fixed_base.xml",
            "scene_fixed_base.xml",
            "sim_scene_fixed_base.xml",
        ):
            p = mjcf_dir / candidate
            if p.is_file():
                return p

    for candidate in ("sim_scene_safe.xml", "scene.xml"):
        p = mjcf_dir / candidate
        if p.is_file():
            if fixed_base:
                try:
                    return ensure_fixed_base_scene(p)
                except Exception:
                    pass
            return p

    raise FileNotFoundError(f"No supported scene XML found in {mjcf_dir}")
