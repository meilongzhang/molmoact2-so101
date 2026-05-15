# MolmoAct2 on SO-101

Zero-shot robot manipulation using [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) from AI2 on a [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm. You give it a natural-language prompt; it figures out the motions. No training, no demonstrations. The model takes RGB frames from a side-view RealSense D455 and a wrist webcam, plus the current joint state, and outputs a chunk of joint targets that get executed with temporal ensembling at 30 Hz. See it in action: [this X post](https://twitter.com/irenekarrot/status/2053966402986647692) — and the official MolmoAct2 announcement from AI2: [allenai.org/blog/molmoact2](https://allenai.org/blog/molmoact2).

---

## Hardware

| Part | Link |
|---|---|
| SO-101 arm (follower) | [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) |
| RealSense D455 (side/scene view, RGB only) | [Intel RealSense D455](https://www.intelrealsense.com/depth-camera-d455/) |
| USB webcam (wrist view) | Any 640×480 USB webcam — we used an XWF-1080P |

The RealSense needs a **USB 3 data cable** (not charge-only) and a USB 3 port. On USB 2.1 it drops to 15 fps which still works but is slower to warm up.

---

## Environment setup

**Python 3.12.** LeRobot 0.5.x requires Python ≥ 3.12 on PyPI, and the 0.5.1 pin below is non-negotiable (see [Calibration](#calibration) for why). Older LeRobot releases ran on 3.10/3.11, but those won't work for this repo.

```bash
conda create -n molmoact python=3.12 -y
conda activate molmoact
```

**Install everything in one shot:**

```bash
pip install -r requirements.txt
```

This installs PyTorch (CUDA 12.1 wheels by default), `lerobot==0.5.1`, the Feetech servo SDK, the HuggingFace stack, the camera libs, and numpy.

**Pins worth knowing:**

| Package | Pin | Why |
|---|---|---|
| `lerobot` | `==0.5.1` | Joint-angle convention — see [Calibration](#calibration). |
| `transformers` | `>=4.52.0` | MolmoAct2's processor imports `transformers.video_utils`, added in 4.52. Earlier versions fail at processor load with `ModuleNotFoundError: No module named 'transformers.video_utils'`. |
| `feetech-servo-sdk` | (any) | SO-101 motor bus. Without it, `inference.py` silently falls back to a simulated follower and the arm will not move. |
| `torch` | CUDA 12.1 wheels | If your driver needs a different CUDA, install torch manually first from [pytorch.org](https://pytorch.org/get-started/locally/), then re-run `pip install -r requirements.txt` (pip will skip the satisfied torch line). |

**MolmoAct2 weights** are downloaded automatically from HuggingFace on first run via `transformers` with `trust_remote_code=True`. No separate install needed.

---

## Calibration

### Joint calibration (required)

LeRobot needs to know each motor's zero position. Run LeRobot's calibration wizard once per arm:

```bash
lerobot-calibrate --robot-type so101 --robot-port /dev/ttyACM0
```

This writes a calibration file to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so_follower.json`.

**Replace `configs/so_follower.json` with your own file** — the one in this repo is specific to our arm and will produce wrong joint angles on yours:

```bash
cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so_follower.json \
   configs/so_follower.json
```

### ⚠️ LeRobot version / calibration gotcha

If you calibrate with LeRobot ≥ 0.5.0 but run inference with the wrong joint-angle offsets, **the arm will slam hard into the table on startup**. This is the most common failure mode.

MolmoAct2 was trained on LeRobot datasets using the v2.1 joint-angle convention. LeRobot 0.5.x records calibrations in the v3.0 convention. The default `--joint-offsets` and `--joint-signs` in `inference.py` apply the official conversion automatically, but only if you calibrated with **exactly lerobot 0.5.1**.

Full details: [https://huggingface.co/docs/lerobot/backwardcomp](https://huggingface.co/docs/lerobot/backwardcomp)

---

## Running

### 1. Find your camera and arm ports

```bash
# List video devices
v4l2-ctl --list-devices

# Find the arm serial port (interactive — unplug/replug to identify)
lerobot-find-port
```

### 2. Launch inference

```bash
python inference.py \
    --follower-port /dev/ttyACM0 \
    --wrist-cam-id 8 \
    --prompt "pick up the lemon and drop it in the red bowl"
```

Add `--show` to open cv2 preview windows for both cameras. Press `Q` or `Ctrl+C` to stop.

### 3. Dry run (model only, arm does not move)

```bash
python inference.py \
    --follower-port /dev/ttyACM0 \
    --wrist-cam-id 8 \
    --prompt "pick up the lemon" \
    --dry-run
```

Use this to verify the model is loading and producing sensible joint targets before letting it move the arm.

### Expected behaviour

- Model downloads on first run (~15 GB). Subsequent runs load from the HF cache.
- Camera warmup takes up to 30 s (RealSense on USB 2.1 is the slow path).
- First inference chunk takes ~700 ms on a laptop GPU (RTX 3080 or better recommended). The arm starts moving after the first chunk; subsequent chunks arrive continuously.
- The console prints inference time, chunk ID, and first/last predicted joint targets each cycle.

### Key options

| Flag | Default | Description |
|---|---|---|
| `--prompt` | *(required)* | Natural-language task instruction |
| `--exec-hz` | `30.0` | Rate at which joint targets are sent to the arm |
| `--max-step-deg` | `15.0` | Per-tick joint motion cap — scale this down if motion is jerky |
| `--num-steps` | `10` | Flow solver iterations — higher = more accurate, slower |
| `--scene-only` | off | Use RealSense image for both inputs, ignore wrist cam |
| `--dry-run` | off | Print actions only, arm does not move |
| `--show` | off | Show cv2 camera preview windows |

---

## Known limitations

- **Wrist webcam is out-of-distribution.** MolmoAct2 was trained on two third-person RealSense views (top + side). The wrist camera view was not in the training data. If the arm behaves poorly, try `--scene-only` to pass the side RealSense view twice and skip the wrist image entirely.
- **No depth into the model.** `enable_depth_reasoning=False` — the RealSense depth stream is not used. RGB only.
- **Single arm only.** This runs the follower arm in autonomous mode with no leader arm or teleop fallback.
- **GPU required.** bfloat16 inference on CUDA. CPU inference is not tested and will be very slow.
- **~15 GB download on first run.** The MolmoAct2 weights are large. Make sure you have disk space and a decent connection.

---

## Troubleshooting

**Arm jerks to one position at startup, then a motor stops responding**
Joint calibration mismatch — see the [calibration gotcha](#️-lerobot-version--calibration-gotcha) above. Power-cycle the arm and re-run `lerobot-calibrate`.

**`No module named 'scservo_sdk'`**
```bash
pip install feetech-servo-sdk
```

**`No RealSense devices found`**
Check `rs-enumerate-devices`. If empty, it's a USB cable or port issue — the D455 needs a USB 3 data cable, not a charge-only cable. Try a different port.

**Wrist camera not found (`/dev/videoN does not exist`)**
Run `v4l2-ctl --list-devices` to find the right index and pass it as `--wrist-cam-id N`.

**`RuntimeError: Cameras did not produce frames in 30s`**
RealSense took too long to warm up. Try a USB 3 port, update D455 firmware with `rs-fw-update -r`, or unplug other USB video devices to free bandwidth.

**Model produces near-zero actions (arm barely moves)**
Auto white balance drift between sessions can cause this. The wrist camera white balance is locked at startup — make sure `v4l2-ctl` is installed (`sudo apt install v4l-utils`). Also try `--scene-only` to rule out wrist image quality as the cause.

**`ImportError: cannot import name 'Teleoperator'`**
Wrong LeRobot version. Pin to exactly 0.5.1:
```bash
pip install --force-reinstall lerobot==0.5.1
```
