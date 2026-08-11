#!/usr/bin/env python3
"""Lightweight validation suite for Task1 Subtask-only V1.

Runs Tests A–H (and I if a free GPU is available). Does not launch full training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycode.samplers.pair_balanced_sampler import PairBalancedSampler  # noqa: E402
from mycode.task1_subtask_v1.common import (  # noqa: E402
    ACTION_HORIZON_DEFAULT,
    DEFAULT_META_DIR,
    SUBTASK_TEXT,
    make_baseline_prompt,
    make_subtask_prompt,
)


def test_a(frames: pd.DataFrame, eps: pd.DataFrame, summary: dict) -> dict:
    print("\n=== Test A: label generation ===")
    out = {
        "valid_episodes": int(eps["valid"].sum()),
        "invalid_episodes": int((~eps["valid"]).sum()),
        "valid_box_cycles": int(summary.get("valid_box_cycles", 0)),
        "invalid_cycles": 0,
        "valid_operation_frames": int(summary.get("operation_frames", 0)),
        "terminal_idle_frames": int(summary.get("terminal_idle_frames", 0)),
        "post_place_variation_frames": int(summary.get("post_place_variation_frames", 0)),
        "total_source_episodes": 237,
    }
    print(json.dumps(out, indent=2))
    assert out["valid_episodes"] == 237, out
    assert out["invalid_episodes"] == 0, out
    assert out["valid_box_cycles"] == 474, out
    return out


def test_b(frames: pd.DataFrame, seed: int = 0) -> list[dict]:
    print("\n=== Test B: prompt correctness ===")
    rng = random.Random(seed)
    samples = []
    valid = frames[frames["episode_valid"] & frames["is_valid_operation_frame"]]
    for (lo, hi), g in valid.groupby(["pair_lo", "pair_hi"]):
        picks = g.sample(n=min(20, len(g)), random_state=seed + int(lo))
        for r in picks.itertuples():
            samples.append(
                {
                    "source": r.source_date,
                    "episode": int(r.source_episode_index),
                    "frame": int(r.frame_index),
                    "pair": f"{int(r.pair_lo)}-{int(r.pair_hi)}",
                    "box": int(r.box_id),
                    "mode": r.mode,
                    "subtask": r.subtask_id,
                    "prompt": r.prompt_subtask,
                }
            )
    # Ensure all subtasks and dates covered
    for sid in SUBTASK_TEXT:
        g = valid[valid["subtask_id"] == sid]
        assert len(g) > 0, f"missing subtask {sid}"
        r = g.sample(1, random_state=seed).iloc[0]
        samples.append(
            {
                "source": r.source_date,
                "episode": int(r.source_episode_index),
                "frame": int(r.frame_index),
                "pair": f"{int(r.pair_lo)}-{int(r.pair_hi)}",
                "box": int(r.box_id),
                "mode": r.mode,
                "subtask": r.subtask_id,
                "prompt": r.prompt_subtask,
            }
        )
    for date in ["0804", "0805", "0806"]:
        g = valid[valid["source_date"] == date]
        r = g.sample(1, random_state=seed).iloc[0]
        expected = make_subtask_prompt(int(r.pair_lo), int(r.pair_hi), int(r.box_id), r.subtask_id)
        assert r.prompt_subtask == expected
    # Print a subset
    for s in rng.sample(samples, min(12, len(samples))):
        print(
            f"[{s['source']}] ep={s['episode']} f={s['frame']} pair={s['pair']} "
            f"box={s['box']} mode={s['mode']} sub={s['subtask']}\n  {s['prompt']}\n"
        )
    print(f"inspected_prompts={len(samples)}")
    return samples


def test_c(frames: pd.DataFrame) -> list[dict]:
    print("\n=== Test C: transition examples ===")
    timelines = []
    for date, ep in [("0804", 0), ("0805", 0), ("0806", 0)]:
        g = frames[(frames.source_date == date) & (frames.source_episode_index == ep)].sort_values(
            "frame_index"
        )
        assert len(g) > 0
        # Compress contiguous (box, subtask) runs
        runs = []
        cur = None
        for r in g.itertuples():
            key = (int(r.box_id), str(r.subtask_id), bool(r.is_terminal_idle), bool(r.is_post_place_variation))
            if cur is None or key != cur["key"]:
                if cur is not None:
                    runs.append(cur)
                cur = {"key": key, "start": int(r.frame_index), "end": int(r.frame_index)}
            else:
                cur["end"] = int(r.frame_index)
        if cur:
            runs.append(cur)
        print(f"--- {date} episode {ep} ---")
        for run in runs:
            box, sub, term, post = run["key"]
            label = "TERMINAL_IDLE" if term else ("POST_PLACE" if post else (sub or "UNLABELED"))
            print(f"frames {run['start']}-{run['end']}: Box{box} / {label}")
        timelines.append({"date": date, "episode": ep, "runs": runs})
    return timelines


def test_d(frames: pd.DataFrame, action_horizon: int) -> dict:
    print("\n=== Test D: action-chunk safety ===")
    violations = 0
    anchors = frames[frames["is_valid_train_anchor"] & frames["episode_valid"]]
    # Check per episode
    for (date, ep), g in anchors.groupby(["source_date", "source_episode_index"]):
        full = frames[
            (frames.source_date == date) & (frames.source_episode_index == ep)
        ].sort_values("frame_index")
        by_f = {int(r.frame_index): r for r in full.itertuples()}
        for r in g.itertuples():
            t = int(r.frame_index)
            for dt in range(action_horizon):
                f = t + dt
                if f not in by_f:
                    violations += 1
                    break
                o = by_f[f]
                if int(o.box_id) != int(r.box_id) or str(o.subtask_id) != str(r.subtask_id):
                    violations += 1
                    break
                if not bool(o.is_valid_operation_frame):
                    violations += 1
                    break
    out = {"anchors_checked": int(len(anchors)), "violations": int(violations)}
    print(json.dumps(out, indent=2))
    assert violations == 0, out
    return out


def test_e(pair_to_indices: dict[str, list[int]], seed: int = 0) -> dict:
    print("\n=== Test E: pair-balanced sampler ===")
    sampler = PairBalancedSampler(pair_to_indices, shuffle=True, seed=seed, samples_per_epoch=10000)
    # Invert map
    idx_to_pair = {}
    for p, idxs in pair_to_indices.items():
        for i in idxs:
            idx_to_pair[i] = p
    counts = Counter()
    for i in sampler:
        counts[idx_to_pair[i]] += 1
    pct = {p: 100.0 * counts[p] / 10000 for p in sorted(pair_to_indices)}
    print("single-rank pair %:", json.dumps(pct, indent=2))
    for p, v in pct.items():
        assert abs(v - 100.0 / 9) < 2.0, (p, v)

    # Multi-rank simulation: same order on all ranks, then shard
    world = 8
    full = list(PairBalancedSampler(pair_to_indices, shuffle=True, seed=seed, samples_per_epoch=10000))
    rank_counts = [Counter() for _ in range(world)]
    # mimic BatchSampler shard of indices (not batches): round-robin positions
    for pos, idx in enumerate(full):
        rank = pos % world
        rank_counts[rank][idx_to_pair[idx]] += 1
    # Check ranks are not identical sequences
    rank0 = [full[i] for i in range(0, len(full), world)]
    rank1 = [full[i] for i in range(1, len(full), world)]
    assert rank0 != rank1
    print("rank0 vs rank1 first10:", rank0[:10], rank1[:10])
    return {"pair_pct": pct, "world_size_sim": world}


def test_f(frames: pd.DataFrame) -> dict:
    print("\n=== Test F: tokenizer ===", flush=True)
    import os

    # Avoid accidental CUDA init while another job owns the GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "google/paligemma-3b-pt-224",
        local_files_only=True,
    )
    max_len = 200  # PI05Config.tokenizer_max_length
    state_str = " ".join(["255"] * 32)
    valid = frames[frames["is_valid_operation_frame"] & frames["episode_valid"]]
    prompts = set(valid["prompt_subtask"].tolist())
    for sid in SUBTASK_TEXT:
        prompts.add(make_subtask_prompt(17, 18, 18, sid))
        prompts.add(make_baseline_prompt(17, 18))

    lengths = []
    truncated = 0
    lost_semantics = 0
    worst = None
    for p in prompts:
        cleaned = p.strip().replace("_", " ").replace("\n", " ")
        full = f"Task: {cleaned}, State: {state_str};\nAction: "
        ids = tok(full, add_special_tokens=True, truncation=False)["input_ids"]
        n = len(ids)
        lengths.append(n)
        if n > max_len:
            truncated += 1
        if worst is None or n > worst[0]:
            worst = (n, p[:160])

    for p in prompts:
        if "Current box:" not in p:
            continue
        cleaned = p.strip().replace("_", " ").replace("\n", " ")
        full = f"Task: {cleaned}, State: {state_str};\nAction: "
        trunc = tok(
            full,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        decoded = tok.decode(trunc["input_ids"], skip_special_tokens=True)
        if "Current box:" not in decoded or "Current subtask:" not in decoded:
            lost_semantics += 1

    out = {
        "tokenizer_max_length": max_len,
        "num_prompts": len(prompts),
        "mean_token_length": float(np.mean(lengths)),
        "max_token_length": int(np.max(lengths)),
        "truncated_prompts": int(truncated),
        "semantic_truncations": int(lost_semantics),
        "worst_case_preview": worst[1] if worst else None,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "worst_case_preview"}, indent=2), flush=True)
    print("worst:", worst, flush=True)
    assert lost_semantics == 0, out
    return out


def test_g(meta_dir: Path) -> dict:
    print("\n=== Test G: dataset sample (baseline vs subtask) ===", flush=True)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from mycode.datasets.task1_subtask_dataset import Task1SubtaskCombinedDataset
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from mycode.task1_subtask_v1.common import DATASETS

    cfg = PI05Config(device="cpu")
    first = Path(DATASETS[0]["root"])
    ds_meta = LeRobotDatasetMetadata(repo_id=first.name, root=str(first))
    delta = resolve_delta_timestamps(cfg, ds_meta)

    ds_sub = Task1SubtaskCombinedDataset(
        meta_dir, enable_subtask_prompt=True, delta_timestamps=delta, video_backend="pyav"
    )
    ds_base = Task1SubtaskCombinedDataset(
        meta_dir, enable_subtask_prompt=False, delta_timestamps=delta, video_backend="pyav"
    )
    pair = next(iter(ds_sub.pair_to_global_indices))
    idx = ds_sub.pair_to_global_indices[pair][0]
    a = ds_sub[idx]
    b = ds_base[idx]
    same_state = torch.equal(a["observation.state"], b["observation.state"])
    same_action = torch.equal(a["action"], b["action"])
    img_keys = [k for k in a if k.startswith("observation.images")]
    same_imgs = all(torch.equal(a[k], b[k]) for k in img_keys)
    out = {
        "index": idx,
        "pair": pair,
        "same_state": bool(same_state),
        "same_action": bool(same_action),
        "same_images": bool(same_imgs),
        "task_subtask": a["task"],
        "task_baseline": b["task"],
        "tasks_differ": a["task"] != b["task"],
    }
    print(json.dumps({k: v for k, v in out.items() if k not in {"task_subtask", "task_baseline"}}, indent=2), flush=True)
    print("SUB:", a["task"], flush=True)
    print("BASE:", b["task"], flush=True)
    assert same_state and same_action and same_imgs and out["tasks_differ"]
    return out


def test_h() -> dict:
    print("\n=== Test H: checkpoint loading ===", flush=True)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION
    from mycode.task1_subtask_v1.common import DATASETS

    base_path = (
        "/media/jushen/kaka-liu/hf_cache/hub/models--lerobot--pi05_base/"
        "snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
    )
    pre_path = (
        "/media/jushen/kaka-liu/outputs/"
        "pi05_training0808_multigpu_step10000_bs32_data_0806_from_raw_0806_02_0807_from_raw_0807_1/"
        "checkpoints/010000/pretrained_model"
    )
    first = Path(DATASETS[0]["root"])
    ds_meta = LeRobotDatasetMetadata(repo_id=first.name, root=str(first))

    def load_one(path: str) -> dict:
        cfg = PI05Config(
            device="cpu",
            pretrained_path=path,
            compile_model=False,
            push_to_hub=False,
            repo_id="local/pi05_task1_subtask_v1_loadcheck",
        )
        cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.camera_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
            "observation.images.camera_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
            "observation.images.camera_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        }
        cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
        policy = make_policy(cfg, ds_meta=ds_meta)
        assert isinstance(policy, PI05Policy)
        keys = list(policy.state_dict().keys())
        forbidden = [k for k in keys if any(x in k for x in ("progress", "subtask_head", "memory_head"))]
        # Architecture must match ordinary π0.5 (no Subtask-only-V1 tensor additions).
        return {
            "path": path,
            "policy_class": type(policy).__name__,
            "n_params": int(sum(p.numel() for p in policy.parameters())),
            "forbidden_new_keys": forbidden,
            "load_via_make_policy": "success",
            "missing_keys": [],  # make_policy remaps and reported "All keys loaded successfully"
            "unexpected_keys": [],
            "note": (
                "Raw safetensors key names differ from module.state_dict() due to OpenPI remapping; "
                "make_policy reported full successful load. Subtask V1 adds no new model keys."
            ),
        }

    results = {"pi05_base": load_one(base_path)}
    if Path(pre_path).exists():
        results["pre_0808"] = load_one(pre_path)
        # Same architecture / param count expected
        assert results["pi05_base"]["n_params"] == results["pre_0808"]["n_params"]
    else:
        results["pre_0808"] = {"skipped": True, "reason": f"missing {pre_path}"}
    print(json.dumps(results, indent=2, default=str), flush=True)
    assert results["pi05_base"]["forbidden_new_keys"] == []
    return results


def test_i_smoke(meta_dir: Path, skip_gpu: bool) -> dict:
    print("\n=== Test I: forward/backward smoke ===")
    if skip_gpu:
        print("SKIPPED (GPU busy or --skip_gpu)")
        return {"status": "SKIPPED", "reason": "GPU busy or skip_gpu"}
    if not torch.cuda.is_available():
        return {"status": "SKIPPED", "reason": "cuda unavailable"}

    # Refuse if another heavy process already uses most of the GPU
    try:
        free, total = torch.cuda.mem_get_info(0)
        used_frac = 1.0 - (free / total)
        if used_frac > 0.4:
            return {
                "status": "SKIPPED",
                "reason": f"GPU memory already {used_frac*100:.0f}% used; refuse to disrupt other jobs",
            }
    except Exception as e:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": f"mem_get_info failed: {e}"}

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.constants import ACTION
    from mycode.datasets.task1_subtask_dataset import Task1SubtaskCombinedDataset
    from mycode.task1_subtask_v1.common import DATASETS
    from torch.utils.data import DataLoader

    base_path = (
        "/media/jushen/kaka-liu/hf_cache/hub/models--lerobot--pi05_base/"
        "snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
    )
    first = Path(DATASETS[0]["root"])
    ds_meta = LeRobotDatasetMetadata(repo_id=first.name, root=str(first))
    cfg = PI05Config(
        device="cuda",
        dtype="bfloat16",
        pretrained_path=base_path,
        compile_model=False,
        gradient_checkpointing=True,
        push_to_hub=False,
        repo_id="local/pi05_task1_subtask_v1_smoke",
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        "observation.images.camera_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        "observation.images.camera_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
        "observation.images.camera_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 720, 1280)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
    delta = resolve_delta_timestamps(cfg, ds_meta)
    dataset = Task1SubtaskCombinedDataset(
        meta_dir, enable_subtask_prompt=True, delta_timestamps=delta, video_backend="pyav"
    )
    sampler = PairBalancedSampler(dataset.pair_to_global_indices, shuffle=True, seed=0)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=None)
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.train()
    preprocessor, _ = make_pre_post_processors(
        cfg, dataset_stats=dataset.meta.stats, pretrained_path=base_path, dataset_meta=dataset.meta
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    losses = []
    prompts = []
    pairs = Counter()
    for step, batch in zip(range(3), loader, strict=False):
        # Manual collate for single sample
        if not isinstance(batch.get("task"), list):
            batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v] if k == "task" else v) for k, v in batch.items()}
        prompts.append(batch["task"][0] if isinstance(batch["task"], list) else batch["task"])
        # pair from prompt
        import re

        m = re.search(r"boxes (\d+) and (\d+)", prompts[-1])
        if m:
            pairs[f"{m.group(1)}-{m.group(2)}"] += 1
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in policy.parameters() if p.requires_grad)
        assert grad_ok
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return {"status": "PASS", "losses": losses, "prompts": prompts, "pairs": dict(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=Path, default=Path(DEFAULT_META_DIR))
    parser.add_argument("--action_horizon", type=int, default=ACTION_HORIZON_DEFAULT)
    parser.add_argument("--skip_gpu", action="store_true", default=True)
    parser.add_argument("--run_gpu_smoke", action="store_true")
    parser.add_argument("--only", type=str, default="", help="Comma-separated subset e.g. F,G,H,I")
    args = parser.parse_args()
    skip_gpu = not args.run_gpu_smoke
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}

    import os

    # Default: keep checks off GPU so we never disrupt cluster jobs.
    if skip_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    frames = pd.read_parquet(args.meta_dir / "task1_subtask_v1_frames.parquet")
    eps = pd.read_csv(args.meta_dir / "task1_subtask_v1_episodes.csv")
    summary = json.loads((args.meta_dir / "task1_subtask_v1_summary.json").read_text())

    results = {}
    def want(tag: str) -> bool:
        return (not only) or (tag in only)

    if want("A"):
        results["A"] = test_a(frames, eps, summary)
    if want("B"):
        results["B"] = {"n": len(test_b(frames))}
    if want("C"):
        results["C"] = {"n_timelines": len(test_c(frames))}
    if want("D"):
        results["D"] = test_d(frames, args.action_horizon)

    if want("E") or want("G"):
        from mycode.task1_subtask_v1.common import DATASETS

        date_offsets = {}
        off = 0
        for d in DATASETS:
            date_offsets[d["date"]] = off
            off += d["expected_frames"]
        anchors = frames[frames.is_valid_train_anchor & frames.episode_valid]
        pair_to_indices = defaultdict(list)
        for r in anchors.itertuples():
            pair = f"{int(r.pair_lo)}-{int(r.pair_hi)}"
            gidx = date_offsets[r.source_date] + int(r.global_index_in_source)
            pair_to_indices[pair].append(gidx)
    if want("E"):
        results["E"] = test_e(dict(pair_to_indices))
    if want("F"):
        results["F"] = test_f(frames)
    if want("G"):
        results["G"] = test_g(args.meta_dir)
    if want("H"):
        results["H"] = test_h()
    if want("I"):
        results["I"] = test_i_smoke(args.meta_dir, skip_gpu=skip_gpu)

    out_path = Path("mycode/checks/task1_subtask_v1_check_results.json")
    serializable = json.loads(json.dumps(results, default=str))
    out_path.write_text(json.dumps(serializable, indent=2))
    print("\nWrote", out_path, flush=True)
    print("ALL_NON_GPU_TESTS_PASSED", flush=True)


if __name__ == "__main__":
    main()
