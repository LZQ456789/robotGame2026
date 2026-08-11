#!/usr/bin/env python3
"""
merge_lerobot_datasets.py
合并若干个本地 LeRobot 数据集为一个数据集。

用法：
    python merge_lerobot_datasets.py \
        --src_roots /media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0806_02 \
                    /media/jushen/kaka-liu/datasets/lerobot_style/0807_from_raw_0807 \
        --output_dir /media/jushen/kaka-liu/datasets/lerobot_style/0806_from_raw_0806_02+0807_from_raw_0807 \
        --output_repo_id nomater

说明：
    - --src_roots 支持传入任意多个（>=2）本地数据集根目录，空格分隔
    - 每个数据集的 repo_id 默认取自它所在文件夹的名字（比如
      .../0802_from_raw_0801 这个目录名就会被当作它的 repo_id），
      不需要额外指定；repo_id 本身在合并这一步只是个内部标识，不影响
      合并逻辑，重要的是 root 指到了正确的本地目录。
    - --output_dir 是合并后数据集实际存放的本地目录，必须指定，
      不指定的话 merge_datasets 会默认存到 $HF_LEROBOT_HOME 下，
      容易和你自己习惯的挂载盘路径规划对不上。
    - 合并前会打印每个源数据集的基本信息（repo_id/总帧数/episode数），
      方便你合并前确认数据源没选错。
    - 合并要求所有源数据集的 features（state/action维度、相机数量等）
      完全一致，如果不一致，merge_datasets 内部会直接报错，报错信息里
      通常会指出具体哪个 feature 不匹配。
"""

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_dataset(root: str) -> LeRobotDataset:
    """从本地目录加载一个 LeRobotDataset，repo_id 直接取目录名。"""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root_path}")
    repo_id = root_path.name
    return LeRobotDataset(repo_id=repo_id, root=str(root_path))


def main():
    parser = argparse.ArgumentParser(description="合并多个本地 LeRobot 数据集")
    parser.add_argument(
        "--src_roots",
        required=True,
        nargs="+",
        help="要合并的若干个本地数据集根目录，空格分隔，至少传2个",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="合并后数据集存放的本地根目录",
    )
    parser.add_argument(
        "--output_repo_id",
        default=None,
        help="合并后数据集的 repo_id（标识名）。不指定时默认取 output_dir 的文件夹名",
    )
    parser.add_argument(
        "--no_concatenate_videos",
        action="store_true",
        help="不合并视频分片，保留每个源文件独立的 mp4（默认会合并成分片）",
    )
    parser.add_argument(
        "--no_concatenate_data",
        action="store_true",
        help="不合并 parquet 分片，保留每个源文件独立的 parquet（默认会合并成分片）",
    )
    args = parser.parse_args()

    if len(args.src_roots) < 2:
        raise ValueError(f"--src_roots 至少需要传入2个数据集根目录，当前只传了 {len(args.src_roots)} 个")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_repo_id = args.output_repo_id or output_dir.name

    print(f"共 {len(args.src_roots)} 个源数据集，准备加载并校验...")
    datasets = []
    for root in args.src_roots:
        ds = load_dataset(root)
        print(f"  - repo_id={ds.repo_id!r}  root={root}  总帧数={len(ds)}  episode数={ds.num_episodes}")
        datasets.append(ds)

    print(f"\n开始合并，输出到: {output_dir}")
    print(f"输出 repo_id: {output_repo_id}")

    merged = merge_datasets(
        datasets,
        output_repo_id=output_repo_id,
        output_dir=str(output_dir),
        concatenate_videos=not args.no_concatenate_videos,
        concatenate_data=not args.no_concatenate_data,
    )

    print("\n合并完成！")
    print(f"合并后总帧数: {len(merged)}")
    print(f"合并后episode数: {merged.num_episodes}")
    print(f"合并后数据集路径: {output_dir}")


if __name__ == "__main__":
    main()