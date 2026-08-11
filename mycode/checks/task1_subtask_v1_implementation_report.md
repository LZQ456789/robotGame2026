# Task1 Subtask-only V1 — Implementation Report

## 1. Files changed

### New (under `mycode/`)

| File | Purpose |
|------|---------|
| `mycode/task1_subtask_v1/common.py` | Shared constants, pair parsing, prompt templates, actuator preprocessing |
| `mycode/task1_subtask_v1/labeling.py` | Offline box-cycle + 4-subtask labeling + action-chunk safety |
| `mycode/preprocessing/build_task1_subtask_v1.py` | Production labeling over 0804/0805/0806 → analysis sidecars |
| `mycode/datasets/task1_subtask_dataset.py` | Multi-source concat dataset; dynamic task string override (Option 1) |
| `mycode/samplers/pair_balanced_sampler.py` | DDP-safe pair-balanced sampler (same seed/epoch contract as EpisodeAwareSampler) |
| `mycode/checks/check_task1_subtask_v1.py` | Tests A–I |
| `mycode/checks/task1_subtask_v1_check_results.json` | Latest check outputs |
| `mycode/train/train_pi05_task1_subtask_v1_base.sh` | Cluster V1 (BASE init, subtask ON) — **not executed** |
| `mycode/train/train_pi05_task1_subtask_v1_control_base.sh` | Cluster CONTROL (BASE init, subtask OFF) — **not executed** |
| `mycode/train/train_pi05_task1_subtask_v1_pre.sh` | Cluster V1 (PRE init, subtask ON) — **not executed** |
| `mycode/checks/task1_subtask_v1_implementation_report.md` | This report |

### Minimal core hooks

| File | Why |
|------|-----|
| `src/lerobot/configs/default.py` | Add `task1_subtask_meta_dir` / `task1_subtask_enable` / `task1_subtask_pair_balance` |
| `src/lerobot/datasets/factory.py` | Route to `Task1SubtaskCombinedDataset` when meta_dir is set |
| `src/lerobot/scripts/lerobot_train.py` | Use `PairBalancedSampler` when Task1 pair-balance is enabled |

### Sidecar outputs (no videos copied)

`/media/jushen/kaka-liu/datasets/analysis/task1_subtask_v1/`

- `task1_subtask_v1_frames.parquet`
- `task1_subtask_v1_episodes.csv`
- `task1_subtask_v1_distribution.csv`
- `task1_subtask_v1_summary.json`

Original 0804/0805/0806 datasets were **not** modified.

---

## 2. Integration architecture

Chosen design: **Option 1 — sidecar + dynamic task override** (least invasive; no video duplication; no π0.5 tensor changes).

```
source LeRobot datasets (0804 / 0805 / 0806)
        │
        ▼
build_task1_subtask_v1.py  →  frames.parquet (labels + prompts + anchor mask)
        │
        ▼
Task1SubtaskCombinedDataset
  - concatenates 3 LeRobotDatasets in-place (reads source videos)
  - __getitem__: load frame → replace item["task"] from sidecar
        │                         (baseline vs subtask via enable flag)
        ▼
PairBalancedSampler (equal P(pair), then uniform valid anchors)
        │
        ▼
standard lerobot-train DataLoader + π0.5 processor/tokenizer
        │
        ▼
--policy.type=pi05   (ordinary PI05Policy; no pi05_progress)
```

Fair baseline: same meta_dir, same sampler, same anchors; only `--dataset.task1_subtask_enable=false` swaps prompts to pair-only text.

---

## 3. Automatic labeling results

Assumptions verified before labeling: fps=26, state/action dim=16, cameras identical, suction=`state[14]`, gripper=`state[15]`, thresholds 0.35 / 0.3.

| Metric | Value |
|--------|------:|
| Source episodes | 237 |
| Valid episodes | 237 |
| Invalid episodes | 0 |
| Valid box cycles | 474 |
| Operation frames | 196,964 |
| Terminal idle frames | 21,246 |
| Post-place variation frames | 6,577 |
| Valid train anchors (after chunk safety) | 137,308 |
| Anchor retention | 69.71% |

By date (operation → anchors):

| Date | Operation | Anchors | Retention |
|------|----------:|--------:|----------:|
| 0804 | 61,851 | 44,146 | 71.4% |
| 0805 | 109,701 | 74,535 | 67.9% |
| 0806 | 25,412 | 18,627 | 73.3% |

Note: ~18 HANDOVER cycles have S2 overlapping the next suction-on (dual-arm). Labels clip S2 at next suction-on so each frame has one `(box_id, subtask_id)`; episodes remain valid.

---

## 4. Action-chunk safety

| Item | Value |
|------|-------|
| π0.5 `chunk_size` / action horizon | **50** |
| `action_delta_indices` | `range(50)` → targets frames `t .. t+49` |
| Episode padding | Dataset clamps + `action_is_pad`; **still rejected as anchors** |
| Rule | Anchor `t` valid iff all of `t..t+49` share same `(box_id, subtask_id)`, are operation frames, and stay in-episode |
| Violations after filtering | **0** / 137,308 anchors |
| Short subtasks unusable (`max_run < 50` & 0 anchors) | **none** |

Retention by subtask:

| Subtask | Operation | Anchors | Retention |
|---------|----------:|--------:|----------:|
| S0_APPROACH_PICK | 53,396 | 32,949 | 61.7% |
| S1_TRANSFER_TO_GRIPPER | 65,808 | 49,785 | 75.7% |
| S2_GRIPPER_PLACEMENT | 44,333 | 28,350 | 63.9% |
| S1_LEFT_DIRECT_PLACE | 33,427 | 26,224 | 78.5% |

---

## 5. Pair-balanced sampling

Simulated 10,000 draws (`seed=0`):

| Pair | % |
|------|--:|
| 1-2 | 11.72 |
| 3-4 | 10.79 |
| 5-6 | 11.15 |
| 7-8 | 11.55 |
| 9-10 | 11.01 |
| 11-12 | 10.90 |
| 13-14 | 10.73 |
| 15-16 | 11.14 |
| 17-18 | 11.01 |

Expected ≈ 11.11% each; all within ±2pp.

Multi-rank simulation (world=8): all ranks share the same `(seed, epoch)` order; accelerate-style sharding yields **disjoint** index streams (rank0 ≠ rank1).

---

## 6. Prompt examples

**S0_APPROACH_PICK**
```
Pack medicine boxes 9 and 10 into the carton.
Current box: 9 of 18.
Current subtask: Approach and pick up the current box with the left suction cup.
```

**S1_TRANSFER_TO_GRIPPER**
```
Pack medicine boxes 1 and 2 into the carton.
Current box: 2 of 18.
Current subtask: Transfer the current box from the left suction cup to the right gripper.
```

**S2_GRIPPER_PLACEMENT**
```
Pack medicine boxes 11 and 12 into the carton.
Current box: 11 of 18.
Current subtask: Place the current box into its target carton position with the right gripper.
```

**S1_LEFT_DIRECT_PLACE**
```
Pack medicine boxes 7 and 8 into the carton.
Current box: 7 of 18.
Current subtask: Place the current box directly into its target carton position with the left suction cup.
```

Baseline control uses only the first line.

Processor note: π0.5 replaces newlines with spaces before tokenization (`Task: …, State: …; Action:`).

---

## 7. Tokenizer validation

Real PaliGemma tokenizer (`google/paligemma-3b-pt-224`, local cache), with full π0.5 prompt including discretized 32-dim state + `Action:`.

| Metric | Value |
|--------|------:|
| `tokenizer_max_length` | 200 |
| Mean token length | 179.36 |
| Max token length | 183 |
| Truncated prompts | **0** |
| Semantic truncations (lost Current box/subtask) | **0** |

---

## 8. Checkpoint compatibility

`make_policy` with `--policy.type=pi05` (no architecture changes):

| Checkpoint | Result |
|------------|--------|
| `pi05_base` snapshot `b211f3d4…` | All keys loaded successfully; `PI05Policy`; 4,143,404,816 params; no progress/subtask heads |
| PRE `…/pi05_training0808_…/checkpoints/010000/pretrained_model` | Same; identical param count |

Missing/unexpected vs raw safetensors names are remapped by OpenPI load path; runtime load reported full success. Subtask-only V1 introduces **no new model keys**.

---

## 9. Smoke-test result

**SKIPPED** — GPU0 already held ~56GB by another job (`nvidia-smi` compute app). Implementation refuses to disrupt that run. Forward/backward smoke can be run later with:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python mycode/checks/check_task1_subtask_v1.py --only I --run_gpu_smoke
```

---

## 10. Fair baseline design

One pipeline, one flag:

| Knob | CONTROL | V1 |
|------|---------|-----|
| Data | 0804+0805+0806 | same |
| Valid-anchor mask | same | same |
| Pair-balanced sampler | same | same |
| `--policy.type` | `pi05` | `pi05` |
| Init / steps / bs / dtype / etc. | same | same |
| `--dataset.task1_subtask_enable` | **false** | **true** |
| Prompt | pair only | pair + current box + current subtask |

Test G confirmed images/state/action identical; only `task` differs.

---

## 11. Cluster commands

**These were NOT executed.**

```bash
# CONTROL (pair-only)
bash mycode/train/train_pi05_task1_subtask_v1_control_base.sh

# V1 (subtask prompts) from pi05_base
bash mycode/train/train_pi05_task1_subtask_v1_base.sh

# Optional V1 from PRE checkpoint
bash mycode/train/train_pi05_task1_subtask_v1_pre.sh
```

Shared settings: `steps=10000`, `batch_size=32`, `dtype=bfloat16`, `gradient_checkpointing=true`, `compile_model=false`, `video_backend=pyav`, multi-node Accelerate. Each script preflight-fails if `OUTPUT_DIR` already exists.

---

## 12. Known risks

1. **Boxes 11–18** have far less raw trajectory diversity (~3 cycles/box vs ~45 for 1–10); pair balancing equalizes sampling probability but does not create new data.
2. Pair balancing prevents 1–10 domination but oversamples scarce 11–18 trajectories → higher overfitting risk on those pairs.
3. Actuator-based subtask labels mark **demonstrated stage transitions**, not placement success verification.
4. Subtask language does **not** directly solve spatial localization / target slot geometry.
5. Action-chunk boundary filtering drops ~30% of operation frames as anchors (especially near subtask ends); remaining context frames still exist for delta targets of earlier anchors.
6. Terminal idle and post-place adjustments are excluded as anchors but remain in episodes for temporal context where needed by earlier chunks.
7. Rare S2 / next-suction overlaps force clipped S2 labels (single-label constraint).

---

## 13. Final verdict

| Gate | Status |
|------|--------|
| Data pipeline | **PASS** |
| Subtask labeling | **PASS** |
| Action-chunk safety | **PASS** |
| Pair-balanced sampler | **PASS** |
| Prompt/tokenizer | **PASS** |
| Checkpoint loading | **PASS** |
| Smoke test | **SKIPPED** (GPU occupied) |

**READY_FOR_CLUSTER_TRAINING**

(Smoke SKIPPED is acceptable per task spec when GPU is unavailable/busy; all other gates pass.)
