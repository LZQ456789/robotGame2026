#!/usr/bin/env python3
"""Prepare Progress-compatible HDF5 inputs for 0806/0807 without modifying raw data.

- Recover box pairs only from reliable evidence (session dir / filename).
- Write mapping CSV + unresolved list.
- Create symlink-based progress_raw directories with `i-j_trajectory.hdf5` names
  so h5_to_lerobotv3.py can auto-generate English pair tasks.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

PAIR_IN_NAME = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})(?!\d)")
VALID_PAIRS = {(i, i + 1) for i in range(1, 18, 2)}


def parse_pair_from_text(text: str) -> tuple[int, int] | None:
    matches = PAIR_IN_NAME.findall(text)
    if not matches:
        return None
    # Prefer the last pair-like token (session names end with i-j).
    first_s, second_s = matches[-1]
    first, second = int(first_s), int(second_s)
    if (first, second) not in VALID_PAIRS:
        return None
    return first, second


def collect_0806_nested(raw_0806: Path) -> tuple[list[dict], list[str]]:
    mapped: list[dict] = []
    unresolved: list[str] = []
    for traj in sorted(raw_0806.rglob("trajectory.hdf5")):
        session = traj.parent.name
        pair = parse_pair_from_text(session)
        if pair is None:
            unresolved.append(str(traj))
            continue
        first, second = pair
        mapped.append(
            {
                "date": "0806",
                "hdf5_path": str(traj.resolve()),
                "first_box": first,
                "second_box": second,
                "source": "session_name",
                "confidence": "high",
                "session": session,
            }
        )
    return mapped, unresolved


def collect_flat_unresolved(raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        return [f"MISSING_RAW_DIR:{raw_dir}"]
    out: list[str] = []
    for p in sorted(raw_dir.glob("*.hdf5")) + sorted(raw_dir.glob("*.h5")):
        if parse_pair_from_text(p.name) is None:
            out.append(str(p.resolve()))
    return out


def collect_recover_unresolved(raw_dir: Path) -> list[str]:
    recover = raw_dir / "recover"
    if not recover.exists():
        return []
    out: list[str] = []
    for traj in sorted(recover.rglob("trajectory.hdf5")):
        # Chinese front/back labels are not 1-2..17-18 pair tasks.
        out.append(str(traj.resolve()))
    return out


def write_mapping_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["date", "hdf5_path", "first_box", "second_box", "source", "confidence"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_mapping(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    seen_paths: set[str] = set()
    for row in rows:
        p = row["hdf5_path"]
        if p in seen_paths:
            issues.append(f"duplicate_mapping_path:{p}")
        seen_paths.add(p)
        first, second = int(row["first_box"]), int(row["second_box"])
        if second < first:
            issues.append(f"second_lt_first:{p}:{first}-{second}")
        if first < 1 or second > 18:
            issues.append(f"out_of_range:{p}:{first}-{second}")
        if (first, second) not in VALID_PAIRS:
            issues.append(f"invalid_pair:{p}:{first}-{second}")
    return issues


def make_symlinks(rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous symlinks only (never delete real files).
    for existing in out_dir.glob("*.hdf5"):
        if existing.is_symlink():
            existing.unlink()
    created: list[Path] = []
    used_names: set[str] = set()
    for row in rows:
        src = Path(row["hdf5_path"])
        session = row.get("session") or src.stem
        # Session already contains i-j (e.g. 0806_test01_1-2); append _trajectory
        # so h5_to_lerobotv3's (...i-j_trajectory.hdf5) regex matches.
        name = f"{session}_trajectory.hdf5"
        # Avoid accidental collisions.
        if name in used_names:
            raise RuntimeError(f"symlink name collision: {name}")
        used_names.add(name)
        dst = out_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        created.append(dst)
    return created


def pair_stats(rows: list[dict], date: str) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        if row["date"] != date:
            continue
        c[f"{row['first_box']}-{row['second_box']}"] += 1
    return dict(sorted(c.items(), key=lambda kv: int(kv[0].split("-")[0])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_0806",
        type=Path,
        default=Path("/media/jushen/kaka-liu/datasets/raw/0806"),
    )
    parser.add_argument(
        "--raw_0806_02",
        type=Path,
        default=Path("/media/jushen/kaka-liu/datasets/raw/0806_02"),
    )
    parser.add_argument(
        "--raw_0807",
        type=Path,
        default=Path("/media/jushen/kaka-liu/datasets/raw/0807"),
    )
    parser.add_argument(
        "--mapping_csv",
        type=Path,
        default=Path(
            "/root/workspace/lerobot/mycode/preprocessing/progress_pair_mapping_0806_0807.csv"
        ),
    )
    parser.add_argument(
        "--unresolved_txt",
        type=Path,
        default=Path("/root/workspace/lerobot/mycode/preprocessing/unresolved_0806_0807.txt"),
    )
    parser.add_argument(
        "--progress_raw_root",
        type=Path,
        default=Path("/media/jushen/kaka-liu/datasets/progress_raw"),
    )
    args = parser.parse_args()

    mapped_0806, unresolved_0806_nested = collect_0806_nested(args.raw_0806)
    unresolved_0806_02 = collect_flat_unresolved(args.raw_0806_02)
    unresolved_0806_02_recover = collect_recover_unresolved(args.raw_0806_02)
    unresolved_0807 = collect_flat_unresolved(args.raw_0807)

    # Only high-confidence mapped rows enter conversion.
    convert_rows = [r for r in mapped_0806 if r["confidence"] == "high"]
    issues = validate_mapping(convert_rows)

    write_mapping_csv(args.mapping_csv, convert_rows)

    unresolved_lines = []
    unresolved_lines.append("# Unresolved HDF5 / missing sources for Progress pair recovery")
    unresolved_lines.append("# 0806 nested without parseable session pair:")
    unresolved_lines.extend(unresolved_0806_nested or ["(none)"])
    unresolved_lines.append("")
    unresolved_lines.append(
        "# 0806_02 flat files used by old 0806_from_raw_0806_02 "
        "(no pair in filename; HDF5 metadata language_instruction=0806_01 only):"
    )
    unresolved_lines.extend(unresolved_0806_02 or ["(none)"])
    unresolved_lines.append("")
    unresolved_lines.append(
        "# 0806_02/recover nested trajectories (正面/反面 labels; not 1-2..17-18 pair tasks):"
    )
    unresolved_lines.extend(unresolved_0806_02_recover or ["(none)"])
    unresolved_lines.append("")
    unresolved_lines.append(
        "# 0807 raw: directory missing on this machine "
        "(old LeRobot 0807_from_raw_0807 exists with 46 eps but no recoverable pair labels)."
    )
    unresolved_lines.extend(unresolved_0807)
    unresolved_lines.append("")
    unresolved_lines.append("# Validation issues on mapped rows:")
    unresolved_lines.extend(issues or ["(none)"])
    args.unresolved_txt.write_text("\n".join(unresolved_lines) + "\n", encoding="utf-8")

    out_0806 = args.progress_raw_root / "0806_pairtask"
    links = make_symlinks(convert_rows, out_0806)

    # Keep empty placeholder dir for 0807 to document intent (no links if unresolved).
    out_0807 = args.progress_raw_root / "0807_pairtask"
    out_0807.mkdir(parents=True, exist_ok=True)
    for existing in out_0807.glob("*.hdf5"):
        if existing.is_symlink():
            existing.unlink()

    print("=== PATH SUMMARY ===")
    print(f"0806 raw path: {args.raw_0806}")
    print(f"0806_02 raw path: {args.raw_0806_02}")
    print(f"0807 raw path: {args.raw_0807} (exists={args.raw_0807.exists()})")
    print(f"mapping csv: {args.mapping_csv}")
    print(f"unresolved: {args.unresolved_txt}")
    print(f"progress_raw 0806: {out_0806} ({len(links)} symlinks)")
    print(f"progress_raw 0807: {out_0807} (0 symlinks)")
    print()
    print("=== 0806 pair distribution (high confidence) ===")
    for k, v in pair_stats(convert_rows, "0806").items():
        print(f"{k}: {v} episodes")
    print(f"total mapped 0806: {len(convert_rows)}")
    print(f"unresolved 0806 nested: {len(unresolved_0806_nested)}")
    print(f"unresolved 0806_02 flat: {len(unresolved_0806_02)}")
    print(f"unresolved 0806_02 recover: {len(unresolved_0806_02_recover)}")
    print(f"unresolved 0807 entries: {len(unresolved_0807)}")
    if issues:
        print("VALIDATION ISSUES:")
        for i in issues:
            print(" ", i)


if __name__ == "__main__":
    main()
