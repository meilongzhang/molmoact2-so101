"""v4l2-ctl helpers for the wrist USB camera.

Single source of truth for the controls toggled on the wrist cam. Calling
reset_wrist_to_auto on entry and exit ensures the camera isn't left in
manual mode for the next consumer.
"""
import subprocess


def set_v4l2(cam_id, **ctrls):
    """Quietly apply a batch of v4l2 controls. Silent no-op if v4l2-ctl is missing."""
    if not ctrls:
        return
    arg = ",".join(f"{k}={v}" for k, v in ctrls.items())
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{cam_id}", "-c", arg],
            check=False, capture_output=True,
        )
    except FileNotFoundError:
        pass


def reset_wrist_to_auto(cam_id):
    """Restore the wrist camera to auto AE / auto WB / brightness=0. Idempotent."""
    set_v4l2(
        cam_id,
        auto_exposure=3,
        white_balance_automatic=1,
        brightness=0,
    )
