# PI0.5 with progress prediction

This policy is an additive extension of `pi05`. It leaves the original policy
unchanged and adds a scalar progress head over the joint image-language prefix
representation.

## Progress supervision

Three target modes are available:

- `task_fraction` (default): for tasks such as
  `Pack medicine boxes 9 and 10 into the carton` or
  `抓取第9个和第10个药盒`, parses the box pair and combines it with episode
  fraction. With `total_task_units=18`, that episode is supervised from
  progress `8/18` to `10/18`.
- Recovery / error episodes must record how many boxes were already placed
  when the failure happened, e.g.
  `Recover after 8 boxes placed: Pack medicine boxes 9 and 10 into the carton`.
  Progress is then anchored at `placed_before` and interpolates to the retry
  pair upper bound.
- `episode_fraction`: automatically uses
  `frame_index / (episode_length - 1)`. This requires no manual labels, but it
  estimates temporal progress within one episode, not verified task success.
- `dataset`: reads a scalar `[0, 1]` target from
  `observation.task_progress` (or the feature selected by `progress_key`). Use
  this when semantic/success-aware labels are available. Keep the
  `observation.` prefix so the standard processor preserves the feature.

The training objective is:

```text
loss = action_flow_matching_loss + progress_loss_weight * smooth_l1_progress_loss
```

## Task text conventions

Normal segment (English preferred for PI0.5):

```text
Pack medicine boxes {i} and {j} into the carton
```

Error / recovery segment (required placed count at failure time):

```text
Recover after {n} boxes placed: Pack medicine boxes {i} and {j} into the carton
```

Chinese equivalents are also parsed:

```text
抓取第{i}个和第{j}个药盒
已放置{n}个药盒后纠错：抓取第{i}个和第{j}个药盒
```

Current plan: full task = **18 boxes**, usually in pairs (`boxes_per_segment=2`).

## Error recovery controller (disabled by default)

`MedicineBoxProgressController` can drive whole-task inference:

1. Feed the current segment prompt from the controller
2. Predict actions + progress each step
3. Advance to the next pair when progress crosses the segment upper bound
4. On an external error signal, switch to a recovery prompt that includes
   `placed_before=n`

Runtime recovery stays **off** until you explicitly enable it:

```python
from lerobot.policies.pi05_progress import (
    ErrorRecoveryConfig,
    MedicineBoxProgressController,
    format_recovery_task,
)

ctrl = MedicineBoxProgressController(
    ErrorRecoveryConfig(
        total_task_units=18,
        enable_error_recovery=False,  # keep False for now
        language="en",
    )
)
ctrl.reset()

# later, once enabled:
# decision = ctrl.signal_error(placed_boxes=8)
# assert "Recover after 8 boxes placed" in decision.task_text
```

Training can already ingest recovery task strings; enabling
`enable_error_recovery` only affects the inference orchestrator.

## Training

The new package is loaded through LeRobot's plugin discovery, so no existing
registry file needs to be changed:

```bash
lerobot-train \
  --policy.discover_packages_path=lerobot.policies.pi05_progress \
  --policy.type=pi05_progress \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.progress_target_mode=task_fraction \
  --policy.total_task_units=18 \
  --policy.enable_error_recovery=false \
  --policy.progress_loss_weight=0.1 \
  --dataset.repo_id=YOUR_DATASET
```

The PI0.5 checkpoint initializes all matching action/VLM weights. The new
progress head is randomly initialized and trained by the auxiliary loss.

## Inference

Normal `select_action` and `predict_action_chunk` behavior is unchanged.
Processed batches can additionally be passed to:

```python
progress = policy.predict_progress(batch)
controller = policy.build_progress_controller()  # recovery still disabled
```

The returned progress tensor is in `[0, 1]`.

For the medicine-box task, `task_fraction` is a zero-annotation bootstrap for
global progress across all 18 boxes. It is still based partly on trajectory
time and must not be treated as proof that a box was correctly placed. A
future `dataset` target based on gripper events or visual success verification
can replace it without changing the model.
