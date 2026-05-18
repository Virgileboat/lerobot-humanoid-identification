from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Linearly rescale delay-log timestamps to a target duration.\n"
            "Assumes timing error is uniformly distributed over the trace."
        )
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default="identification_2/results",
        help="Directory containing delay CSV logs.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="delay_all_*.csv",
        help="Glob pattern for input CSV logs (inside --input-dir).",
    )
    p.add_argument(
        "--target-duration-s",
        type=float,
        default=2.0,
        help="Target trace duration in seconds after rescaling.",
    )
    p.add_argument(
        "--in-place",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite input files directly (with backups).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory when not in-place.",
    )
    p.add_argument(
        "--backup-dir",
        type=str,
        default="",
        help="Backup directory used when --in-place is true.",
    )
    return p.parse_args()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: missing header.")
        fieldnames = list(reader.fieldnames)
        if "t_s" not in fieldnames:
            raise RuntimeError(f"{path}: missing required column 't_s'.")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{path}: empty file.")
    return fieldnames, rows


def _rescale_rows(rows: list[dict[str, str]], target_duration_s: float) -> tuple[list[dict[str, str]], float, float]:
    t_vals = [float(r["t_s"]) for r in rows]
    t0 = float(t_vals[0])
    t1 = float(t_vals[-1])
    span = t1 - t0
    if not math.isfinite(span) or span <= 1e-12:
        raise RuntimeError(f"invalid span: t0={t0}, t1={t1}, span={span}")

    scale = float(target_duration_s) / span
    out_rows: list[dict[str, str]] = []
    for r, t in zip(rows, t_vals):
        r2 = dict(r)
        t_new = (float(t) - t0) * scale
        r2["t_s"] = f"{t_new:.9f}"
        out_rows.append(r2)
    return out_rows, span, scale


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = _parse_args()
    if args.target_duration_s <= 0.0:
        raise ValueError("--target-duration-s must be > 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    in_paths = sorted(input_dir.glob(str(args.glob)))
    if not in_paths:
        raise RuntimeError(f"No files found in {input_dir} matching glob: {args.glob}")

    in_place = bool(args.in_place)
    output_dir: Path | None = None
    backup_dir: Path | None = None
    if in_place:
        if args.backup_dir.strip():
            backup_dir = Path(args.backup_dir).expanduser().resolve()
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = input_dir / f"_backup_before_rescale_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[setup] backup_dir={backup_dir}")
    else:
        if not args.output_dir.strip():
            raise ValueError("--output-dir is required when --in-place is false")
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[setup] output_dir={output_dir}")

    print(f"[setup] input_dir={input_dir}")
    print(f"[setup] glob={args.glob}")
    print(f"[setup] target_duration_s={float(args.target_duration_s):.6f}")
    print(f"[setup] files={len(in_paths)}")

    count = 0
    for in_path in in_paths:
        fieldnames, rows = _read_rows(in_path)
        out_rows, old_span, scale = _rescale_rows(rows, float(args.target_duration_s))

        if in_place:
            assert backup_dir is not None
            backup_path = backup_dir / in_path.name
            shutil.copy2(in_path, backup_path)
            out_path = in_path
        else:
            assert output_dir is not None
            out_path = output_dir / in_path.name

        _write_rows(out_path, fieldnames, out_rows)
        count += 1
        print(
            f"[ok] {in_path.name}: old_span={old_span:.6f}s -> "
            f"new_span={float(args.target_duration_s):.6f}s (scale={scale:.6f})"
        )

    print(f"[done] rescaled files: {count}")


if __name__ == "__main__":
    main()
