#!/usr/bin/env python3
"""Convert Task2 nested HDF5 trajectories to LeRobot v3 datasets.

Task2 raw layout (do NOT flatten / move originals):

    <date_root>/
      test_01/<session>/trajectory.hdf5
      test_02/<session>/trajectory.hdf5
      ...

Each trajectory.hdf5 is one complete Task2 demonstration and becomes one
LeRobot episode. Episodes under the same test_xx are repeated demos of the
same task (not different semantic tasks).

Reuses verified HDF5→LeRobot conversion logic from h5_to_lerobotv3.py
(state/action/cameras/fps/video encoding). Adds:
  - recursive discovery of test_xx/session/trajectory.hdf5
  - fixed Task2 language instruction for all episodes
  - episode→source HDF5 mapping CSV

Usage:
    uv run python mycode/preprocessing/h5_to_lerobot_task2.py \\
        --src_dir /media/jushen/kaka-liu/datasets/raw/20260808 \\
        --repo_id nomater \\
        --root /media/jushen/kaka-liu/datasets/lerobot_style/task2_from_raw_0808 \\
        --date_tag 0808 \\
        --mapping_csv /root/workspace/lerobot/mycode/preprocessing/task2_0808_episode_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Ensure sibling import works when launched as a script path.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse Task1-validated conversion helpers (do not change observation/action schema).
from h5_to_lerobotv3 import (  # noqa: E402
    ACTION_FIELDS,
    CAMERA_GROUP,
    CAMERA_KEYS,
    STATE_FIELDS,
    build_dim_names,
    build_features,
    check_required_fields,
    concat_frame_from_bulk,
    decode_all_cameras,
    decode_image,
    estimate_fps,
    find_schema_probe_file,
    get_num_frames,
    read_fields_bulk,
)

DEFAULT_TASK2_INSTRUCTION = (
    "Place the medicine and instruction leaflet into the medicine box, "
    "then place the medicine box into the carton."
)

TEST_DIR_RE = re.compile(r"^test_(\d+)$")
SESSION_TRAIL_1_2_RE = re.compile(r"^(.+)_([12])$")
SESSION_NUM_RE = re.compile(r"_(\d+)$")


def parse_test_id(name: str) -> int | None:
    m = TEST_DIR_RE.fullmatch(name)
    return int(m.group(1)) if m else None


def parse_session_number(session_name: str) -> int | None:
    m = SESSION_NUM_RE.search(session_name)
    return int(m.group(1)) if m else None


def parse_trial_order_from_name(session_name: str) -> int | None:
    """Only accept an exact trailing _1 / _2 (not _12, _21, etc.)."""
    m = SESSION_TRAIL_1_2_RE.fullmatch(session_name)
    if not m:
        return None
    return int(m.group(2))


def discover_task2_trajectories(src_dir: str | Path) -> list[dict]:
    """Find all test_xx/<session>/trajectory.hdf5 under src_dir.

    Ordering: test_id ascending, then session number ascending (stable episode order).
    """
    root = Path(src_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"src_dir 不存在: {root}")

    rows: list[dict] = []
    for test_dir in sorted(root.iterdir()):
        if not test_dir.is_dir():
            continue
        test_num = parse_test_id(test_dir.name)
        if test_num is None:
            continue

        sessions = [p for p in test_dir.iterdir() if p.is_dir()]
        sessions.sort(
            key=lambda p: (
                parse_session_number(p.name) is None,
                parse_session_number(p.name) if parse_session_number(p.name) is not None else 10**12,
                p.name,
            )
        )

        # Within each test, sorted session order defines trial 1/2 when name suffix is not _1/_2.
        for within_test_idx, session_dir in enumerate(sessions, start=1):
            traj = session_dir / "trajectory.hdf5"
            if not traj.is_file():
                # Also accept *.h5 named trajectory.h5 if present.
                alt = session_dir / "trajectory.h5"
                if alt.is_file():
                    traj = alt
                else:
                    continue

            name_order = parse_trial_order_from_name(session_dir.name)
            if name_order is not None:
                trial_order = name_order
                trial_order_source = "session_name_suffix"
            else:
                trial_order = within_test_idx
                trial_order_source = "within_test_session_sort"

            rows.append(
                {
                    "test_id": test_dir.name,
                    "test_num": test_num,
                    "session_name": session_dir.name,
                    "trial_order": trial_order,
                    "trial_order_source": trial_order_source,
                    "source_hdf5": str(traj.resolve()),
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"{root} 下没有找到 test_xx/<session>/trajectory.hdf5"
        )
    return rows


def write_mapping_csv(path: Path, date_tag: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_index",
        "date",
        "test_id",
        "trial_order",
        "session_name",
        "source_hdf5",
        "trial_order_source",
        "status",
        "n_frames",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "episode_index": row.get("episode_index", ""),
                    "date": date_tag,
                    "test_id": row["test_id"],
                    "trial_order": row.get("trial_order", ""),
                    "session_name": row["session_name"],
                    "source_hdf5": row["source_hdf5"],
                    "trial_order_source": row.get("trial_order_source", ""),
                    "status": row.get("status", ""),
                    "n_frames": row.get("n_frames", ""),
                    "error": row.get("error", ""),
                }
            )


def convert_task2(
    src_dir: str,
    repo_id: str,
    root: str,
    robot_type: str,
    date_tag: str,
    mapping_csv: str,
    num_decode_workers: int,
    image_writer_processes: int,
    image_writer_threads: int,
    task_description: str,
):
    out_root = Path(root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(
            f"输出目录已存在且非空，拒绝覆盖: {out_root}\n"
            "请换一个新的 --root，或先手动清理。"
        )

    discovered = discover_task2_trajectories(src_dir)
    h5_files = [r["source_hdf5"] for r in discovered]
    print(f"共发现 {len(discovered)} 个 Task2 trajectory.hdf5")
    print(f"输出: {out_root}")
    print(f"固定 Task2 instruction: 「{task_description}」")

    suffix_count = sum(1 for r in discovered if r["trial_order_source"] == "session_name_suffix")
    sort_count = len(discovered) - suffix_count
    print(
        f"trial_order: {suffix_count} 条来自 session 名末尾 _1/_2；"
        f"{sort_count} 条因命名为全局序号，改用同 test 内 session 编号排序 (1..N)"
    )
    if sort_count:
        examples = [
            f"{r['test_id']}/{r['session_name']}→{r['trial_order']}"
            for r in discovered
            if r["trial_order_source"] == "within_test_session_sort"
        ][:8]
        print(f"  排序推导示例: {examples}")

    probe_path = find_schema_probe_file(h5_files)
    print(f"使用 {probe_path} 探测 schema")

    with h5py.File(probe_path, "r") as f:
        if not check_required_fields(f):
            raise RuntimeError(f"探测文件字段不全: {probe_path}")
        state_bulk0 = read_fields_bulk(f, STATE_FIELDS)
        action_bulk0 = read_fields_bulk(f, ACTION_FIELDS)
        state0 = concat_frame_from_bulk(state_bulk0, STATE_FIELDS, 0)
        action0 = concat_frame_from_bulk(action_bulk0, ACTION_FIELDS, 0)
        state_names = build_dim_names(f, STATE_FIELDS)
        action_names = build_dim_names(f, ACTION_FIELDS)
        img0 = decode_image(f[f"{CAMERA_GROUP}/{CAMERA_KEYS[0]}"][0])
        img_shape = img0.shape
        fps = estimate_fps(f)

    print(
        f"探测到 state维度={state0.shape[0]}, action维度={action0.shape[0]}, "
        f"图像尺寸={img_shape}, fps≈{fps}"
    )

    features = build_features(
        state0.shape[0], action0.shape[0], img_shape, state_names, action_names
    )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=str(out_root),
        features=features,
        robot_type=robot_type,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    n_success, n_skipped = 0, 0
    mapping_rows: list[dict] = []
    decode_executor = ThreadPoolExecutor(max_workers=num_decode_workers)
    episode_index = 0

    try:
        for fi, meta in enumerate(discovered):
            h5_path = meta["source_hdf5"]
            print(
                f"[{fi + 1}/{len(discovered)}] ep{episode_index} "
                f"{meta['test_id']}/{meta['session_name']} ← {h5_path}"
            )
            row = dict(meta)

            try:
                with h5py.File(h5_path, "r") as f:
                    n_frames = get_num_frames(f)

                    # Length sanity: refuse silent truncation.
                    lengths = {}
                    for p in STATE_FIELDS + ACTION_FIELDS:
                        lengths[p] = f[p].shape[0]
                    for cam in CAMERA_KEYS:
                        lengths[f"{CAMERA_GROUP}/{cam}"] = f[f"{CAMERA_GROUP}/{cam}"].shape[0]
                    if "camera_observations/timestamp" in f:
                        lengths["camera_observations/timestamp"] = f[
                            "camera_observations/timestamp"
                        ].shape[0]
                    unique_lens = set(lengths.values())
                    if len(unique_lens) != 1:
                        raise ValueError(
                            "frame length mismatch: "
                            + ", ".join(f"{k}={v}" for k, v in lengths.items())
                        )

                    state_bulk = read_fields_bulk(f, STATE_FIELDS)
                    action_bulk = read_fields_bulk(f, ACTION_FIELDS)
                    cam_datasets = {cam: f[f"{CAMERA_GROUP}/{cam}"] for cam in CAMERA_KEYS}

                    for t in range(n_frames):
                        frame = {
                            "observation.state": concat_frame_from_bulk(
                                state_bulk, STATE_FIELDS, t
                            ),
                            "action": concat_frame_from_bulk(
                                action_bulk, ACTION_FIELDS, t
                            ),
                            "task": task_description,
                        }
                        images = decode_all_cameras(cam_datasets, t, decode_executor)
                        for cam, img in images.items():
                            frame[f"observation.images.{cam}"] = img
                        dataset.add_frame(frame)
            except Exception as e:
                print(f"  [跳过] 读取该文件失败: {e}")
                n_skipped += 1
                row.update(
                    {
                        "episode_index": "",
                        "status": "failed_read",
                        "n_frames": "",
                        "error": str(e),
                    }
                )
                mapping_rows.append(row)
                if hasattr(dataset, "clear_episode_buffer"):
                    dataset.clear_episode_buffer()
                continue

            try:
                dataset.save_episode()
            except Exception as e:
                print(f"\n[致命错误] episode 保存/视频编码失败: {e}")
                row.update(
                    {
                        "episode_index": "",
                        "status": "failed_save",
                        "n_frames": n_frames,
                        "error": str(e),
                    }
                )
                mapping_rows.append(row)
                write_mapping_csv(Path(mapping_csv), date_tag, mapping_rows)
                raise

            row.update(
                {
                    "episode_index": episode_index,
                    "status": "ok",
                    "n_frames": n_frames,
                    "error": "",
                }
            )
            mapping_rows.append(row)
            episode_index += 1
            n_success += 1

            # Incremental mapping after every successful episode (crash-safe + progress).
            write_mapping_csv(Path(mapping_csv), date_tag, mapping_rows)
            print(
                f"  saved episode_index={episode_index - 1} n_frames={n_frames} "
                f"(success={n_success}/{len(discovered)})",
                flush=True,
            )
    finally:
        decode_executor.shutdown(wait=True)

    dataset.finalize()
    write_mapping_csv(Path(mapping_csv), date_tag, mapping_rows)

    # Also copy mapping next to the dataset for convenience.
    side_csv = out_root / f"task2_{date_tag}_episode_mapping.csv"
    write_mapping_csv(side_csv, date_tag, mapping_rows)

    print("\n转换结束。")
    print(f"成功: {n_success} 个 | 跳过: {n_skipped} 个 | 共 {len(discovered)} 个文件")
    print(f"输出目录: {out_root}")
    print(f"mapping CSV: {mapping_csv}")
    print(f"mapping CSV (dataset side): {side_csv}")
    if n_skipped:
        print("\n⚠️ 失败文件：")
        for row in mapping_rows:
            if row.get("status") != "ok":
                print(f"  - {row['source_hdf5']}: {row.get('error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Task2 nested HDF5 → LeRobot v3.0（每 trajectory.hdf5 = 1 episode）"
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Task2 原始日期根目录，如 .../raw/20260808（含 test_xx/）",
    )
    parser.add_argument("--repo_id", required=True, help="数据集逻辑名")
    parser.add_argument(
        "--root",
        required=True,
        help="输出数据集目录（最后一级必须尚不存在或为空）",
    )
    parser.add_argument("--robot_type", default="humanoid_wam")
    parser.add_argument(
        "--date_tag",
        required=True,
        help="写入 mapping 的日期标签，如 0808 / 0809",
    )
    parser.add_argument(
        "--mapping_csv",
        required=True,
        help="episode→source HDF5 映射 CSV 路径",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=DEFAULT_TASK2_INSTRUCTION,
        help="所有 episode 共用的 Task2 语言指令",
    )
    parser.add_argument(
        "--num_decode_workers",
        type=int,
        default=max(os.cpu_count() or 4, 4),
    )
    parser.add_argument("--image_writer_processes", type=int, default=4)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    args = parser.parse_args()

    convert_task2(
        src_dir=args.src_dir,
        repo_id=args.repo_id,
        root=args.root,
        robot_type=args.robot_type,
        date_tag=args.date_tag,
        mapping_csv=args.mapping_csv,
        num_decode_workers=args.num_decode_workers,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        task_description=args.task_description,
    )


if __name__ == "__main__":
    main()
