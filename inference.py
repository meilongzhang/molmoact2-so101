"""
inference.py — Run MolmoAct2-SO100_101 zero-shot on the SO-101.

Single machine, no WebRTC. Captures the wrist USB camera and the RealSense
D455 colour stream, feeds both plus the current joint state to MolmoAct2, and
executes the predicted action chunks on the follower arm at --exec-hz via
temporal ensembling.

Usage:
    python inference.py \
        --follower-port /dev/ttyACM0 \
        --wrist-cam-id 8 \
        --prompt "pick up the lemon and drop it in the red bowl"
"""
import argparse
import atexit
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from molmoact_so101.setup.robot import FollowerArm, RealSenseCapture
from molmoact_so101.setup.wrist_camera import WristCamera, FLIP_CHOICES
from molmoact_so101.setup.frame_transforms import (
    parse_joint_limits, parse_joint_offsets, parse_joint_signs,
)
from molmoact_so101.model.policy import MolmoActPolicy, REPO_ID, DTYPES as _DTYPES
from molmoact_so101.model.runtime import AsyncPolicyRunner, RuntimeConfig


def parse_args():
    p = argparse.ArgumentParser(description="MolmoAct2 zero-shot inference on SO-101")
    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--follower-port", default="/dev/ttyACM0",
                   help="Serial port for the SO-101 follower arm.")
    p.add_argument("--wrist-cam-id", type=int, default=8,
                   help="OpenCV index of the wrist USB camera. "
                        "Find yours with `v4l2-ctl --list-devices`.")
    p.add_argument("--wrist-flip", choices=FLIP_CHOICES, default="180",
                   help="Flip the wrist image to match training orientation.")
    p.add_argument("--realsense-serial", default=None,
                   help="RealSense D455 serial number. Omit to use the first device found.")
    # ── Task ─────────────────────────────────────────────────────────────────
    p.add_argument("--prompt", required=True,
                   help="Natural-language task instruction, e.g. "
                        "'pick up the lemon and drop it in the red bowl'.")
    # ── Inference ─────────────────────────────────────────────────────────────
    p.add_argument("--num-steps", type=int, default=10,
                   help="MolmoAct2 continuous-flow solver iterations.")
    p.add_argument("--actions-per-chunk", type=int, default=None,
                   help="Execute up to this many steps per chunk before re-querying "
                        "the model. Default: full chunk (~30 steps at 30 fps).")
    p.add_argument("--exec-hz", type=float, default=30.0,
                   help="Rate at which joint targets are sent to the arm (Hz).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    p.add_argument("--cuda-graph", action="store_true",
                   help="Enable cuda graphs (faster, uses more VRAM, needs a warm-up).")
    p.add_argument("--warmup-predictions", type=int, default=None,
                   help="Discard this many initial predictions before moving the arm. "
                        "Defaults to 2 with --cuda-graph, else 0.")
    # ── Safety ────────────────────────────────────────────────────────────────
    p.add_argument("--max-step-deg", type=float, default=15.0,
                   help="Per-tick joint motion cap (degrees). The entire delta is "
                        "scaled down if any joint would exceed this limit.")
    p.add_argument("--joint-min", default=None,
                   help="Per-joint hard floor in arm frame (degrees). "
                        "Comma-separated 6 values; use 'none' to skip a joint. "
                        "Example: 'none,-65,none,none,none,none'")
    p.add_argument("--joint-max", default=None,
                   help="Per-joint hard ceiling in arm frame (degrees). Same format.")
    # ── Frame transform ───────────────────────────────────────────────────────
    p.add_argument("--joint-offsets", default="0,90,90,0,0,0",
                   help="Per-joint offset (deg) added when converting arm frame → "
                        "model frame. Default is the official LeRobot v3.0→v2.1 "
                        "SO-100/101 conversion. See --joint-signs.")
    p.add_argument("--joint-signs", default="1,-1,1,1,1,1",
                   help="Per-joint sign multiplier (±1). Default is the official "
                        "LeRobot v3.0→v2.1 conversion (sign-flip on shoulder_lift). "
                        "See https://huggingface.co/docs/lerobot/backwardcomp")
    # ── Ensembling / smoothing ────────────────────────────────────────────────
    p.add_argument("--ensemble-m", type=float, default=0.5,
                   help="Temporal-ensembling decay (1/s). Lower = older chunks "
                        "contribute more (smoother, more lag).")
    p.add_argument("--smooth-alpha", type=float, default=1.0,
                   help="Final EMA low-pass on ensemble output (1.0 = off). "
                        "Try 0.5–0.7 if you see high-frequency jitter.")
    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--scene-only", action="store_true",
                   help="Pass the RealSense scene image twice, ignoring the wrist "
                        "camera. The training data uses two third-person views; "
                        "try this if wrist images are out-of-distribution.")
    p.add_argument("--save-frames-dir", default=None,
                   help="Save model input images to this directory each cycle "
                        "(useful for debugging what the model sees).")
    p.add_argument("--no-wrist-ae", action="store_true",
                   help="Disable adaptive wrist-camera brightness controller.")
    p.add_argument("--show", action="store_true",
                   help="Show cv2 camera preview windows (press Q to quit).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run inference and print actions but do NOT move the arm.")
    return p.parse_args()


def warmup_cameras(wrist: WristCamera, scene: RealSenseCapture,
                   timeout: float = 30.0) -> None:
    """Block until both cameras produce frames, or raise on timeout.

    RealSense on USB 2.1 can take several seconds to deliver the first frame.
    """
    print("[MolmoAct] Warming up cameras (up to 30 s)...")
    t_start  = time.time()
    next_log = t_start + 2.0
    wrist_ok = scene_ok = False
    while time.time() - t_start < timeout:
        if not wrist_ok and wrist.read() is not None:
            wrist_ok = True
            print(f"[MolmoAct]   wrist ready ({time.time()-t_start:.1f}s)")
        if not scene_ok and scene.get_latest_color() is not None:
            scene_ok = True
            print(f"[MolmoAct]   RealSense ready ({time.time()-t_start:.1f}s)")
        if wrist_ok and scene_ok:
            return
        if time.time() > next_log:
            print(f"[MolmoAct]   waiting... wrist={wrist_ok} realsense={scene_ok}")
            next_log = time.time() + 2.0
        time.sleep(0.1)
    raise RuntimeError(
        f"Cameras did not produce frames in {timeout:.0f}s "
        f"(wrist={wrist_ok}, realsense={scene_ok}). "
        "Check USB connections — RealSense needs a USB-3 data cable."
    )


def install_cleanup_handlers(follower: FollowerArm, wrist: WristCamera,
                             scene: RealSenseCapture):
    """Register an idempotent cleanup that fires on any exit path."""
    lock = threading.Lock()
    done = [False]

    def cleanup(*_):
        if not lock.acquire(blocking=False):
            return
        if done[0]:
            return
        done[0] = True
        print("\n[MolmoAct] cleanup: disabling torque + releasing devices")
        for fn in [
            lambda: follower.request_torque(False),
            lambda: follower.disconnect(),
            lambda: scene.release(),
            lambda: wrist.close(),
        ]:
            try:
                fn()
            except Exception:
                pass

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *a: sys.exit(0))
    return cleanup


def run_display_loop(runner: AsyncPolicyRunner, show: bool) -> None:
    """Main-thread loop: optionally show camera windows, otherwise idle."""
    try:
        while True:
            if show:
                wrist_bgr, scene_bgr = runner.latest_frames()
                if wrist_bgr is not None:
                    cv2.imshow("wrist", wrist_bgr)
                if scene_bgr is not None:
                    cv2.imshow("realsense", scene_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[MolmoAct] Interrupted.")


def main():
    args = parse_args()
    if args.warmup_predictions is None:
        args.warmup_predictions = 2 if args.cuda_graph else 0

    joint_offsets = parse_joint_offsets(args.joint_offsets)
    joint_signs   = parse_joint_signs(args.joint_signs)
    joint_min     = parse_joint_limits(args.joint_min, -np.inf)
    joint_max     = parse_joint_limits(args.joint_max,  np.inf)

    if np.any(joint_offsets != 0) or np.any(joint_signs != 1):
        print("[MolmoAct] Frame transform: state→model = signs * arm_state + offsets")
        print(f"           signs   = {joint_signs.tolist()}")
        print(f"           offsets = {joint_offsets.tolist()}")

    policy = MolmoActPolicy.from_pretrained(
        REPO_ID,
        dtype=args.dtype,
        device=args.device,
        apply_patches=False,
    )

    wrist    = WristCamera(args.wrist_cam_id, flip=args.wrist_flip,
                           enable_ae=not args.no_wrist_ae)
    follower = FollowerArm(port=args.follower_port)
    scene    = RealSenseCapture(serial=args.realsense_serial)

    warmup_cameras(wrist, scene)
    follower.set_target(follower.get_state())  # latch current pose before torque-on
    cleanup = install_cleanup_handlers(follower, wrist, scene)

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)
        print(f"[MolmoAct] Saving model-input frames to {args.save_frames_dir}")

    print(f"[MolmoAct] Task: {args.prompt!r}")
    if args.dry_run:
        print("[MolmoAct] --dry-run: arm will NOT move.")
    print("[MolmoAct] Press Ctrl+C to stop.")

    config = RuntimeConfig(
        prompt=args.prompt,
        exec_hz=args.exec_hz,
        max_step_deg=args.max_step_deg,
        actions_per_chunk=args.actions_per_chunk,
        smooth_alpha=args.smooth_alpha,
        ensemble_m=args.ensemble_m,
        warmup_predictions=args.warmup_predictions,
        num_steps=args.num_steps,
        cuda_graph=args.cuda_graph,
        scene_only=args.scene_only,
        save_frames_dir=args.save_frames_dir,
        dry_run=args.dry_run,
    )

    try:
        with AsyncPolicyRunner(
            policy=policy, follower=follower, wrist=wrist, scene=scene,
            signs=joint_signs, offsets=joint_offsets,
            joint_min=joint_min, joint_max=joint_max,
            config=config,
        ) as runner:
            run_display_loop(runner, args.show)
    finally:
        cleanup()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
