"""
robot.py — SO-101 follower arm and RealSense D455 camera.

FollowerArm wraps the LeRobot SOFollower driver. All serial I/O runs in a
background thread; callers use set_target() / get_state() without blocking.

RealSenseCapture opens one pyrealsense2 pipeline for colour (RGB) at 640×480.
A background thread continuously pulls frames; callers use get_latest_color()
for non-blocking access. Depth is captured but not forwarded to the model
(MolmoAct2 uses enable_depth_reasoning=False).
"""
import threading
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np


JOINT_COUNT = 6
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
_MAX_JOINT_DELTA = 10.0  # degrees per worker iteration — inner rate limiter


class FollowerArm:
    """Drives the SO-101 follower arm via LeRobot's SOFollower driver.

    All serial I/O is handled by a single background worker thread that
    alternates write / read on the half-duplex Feetech bus. Call set_target()
    and get_state() from any thread; they touch only lock-protected slots.
    """

    def __init__(self, port: str = "/dev/ttyACM0", simulate: bool = False):
        self.simulate = simulate
        self._target = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._state  = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._target_lock  = threading.Lock()
        self._state_lock   = threading.Lock()
        self._torque_lock  = threading.Lock()
        self._torque_desired = True
        self._stop   = threading.Event()
        self._thread = None

        if not simulate:
            try:
                from lerobot.robots.so_follower import SOFollower
                from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
                self.robot = SOFollower(SOFollowerRobotConfig(
                    port=port, id="so_follower", use_degrees=True
                ))
                self.robot.connect()
                print(f"[Follower] Connected on {port}")
                obs  = self.robot.get_observation()
                init = np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=np.float32)
                self._target = init.copy()
                self._state  = init.copy()
                self._thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._thread.start()
            except Exception as e:
                print(f"[Follower] Could not connect: {e} — falling back to simulation")
                self.simulate = True
        else:
            print("[Follower] Simulation mode")

    def _worker_loop(self):
        torque_actual = True
        last_written  = self._target.copy()
        while not self._stop.is_set():
            with self._torque_lock:
                desired = self._torque_desired
            if desired != torque_actual:
                try:
                    if desired:
                        self.robot.bus.enable_torque()
                        print("[Follower] Torque enabled")
                    else:
                        self.robot.bus.disable_torque()
                        print("[Follower] Torque disabled")
                    torque_actual = desired
                except Exception as e:
                    print(f"[Follower] torque transition error: {e}")

            if torque_actual:
                with self._target_lock:
                    target = self._target.copy()
                delta = np.clip(target - last_written, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
                last_written = last_written + delta
                try:
                    action = {f"{n}.pos": float(last_written[i])
                              for i, n in enumerate(MOTOR_NAMES)}
                    self.robot.send_action(action)
                except Exception as e:
                    print(f"[Follower] send_action error: {e}")

            try:
                obs   = self.robot.get_observation()
                state = np.array([obs[f"{n}.pos"] for n in MOTOR_NAMES], dtype=np.float32)
                with self._state_lock:
                    self._state = state
            except Exception as e:
                print(f"[Follower] get_observation error: {e}")

    def set_target(self, target: np.ndarray):
        if self.simulate:
            with self._target_lock:
                delta = np.clip(target - self._target, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
                self._target = self._target + delta
            with self._state_lock:
                self._state = self._target.copy()
            return
        with self._target_lock:
            self._target = target.astype(np.float32, copy=True)

    def get_state(self) -> np.ndarray:
        with self._state_lock:
            return self._state.copy()

    def request_torque(self, on: bool):
        if self.simulate:
            return
        with self._torque_lock:
            self._torque_desired = on

    def disconnect(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.simulate:
            try:
                self.robot.bus.disable_torque()
            except Exception:
                pass
            self.robot.disconnect()


class SceneCaptureBase(ABC):
    """Base class for scene cameras used by AsyncPolicyRunner.

    Subclasses must return BGR uint8 frames from _read_frame().
    """

    name = "SceneCapture"

    def __init__(self):
        self._color = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @abstractmethod
    def _read_frame(self):
        """Return latest BGR frame, or None if unavailable."""
        pass

    @abstractmethod
    def _stop_backend(self):
        """Release camera-specific resources."""
        pass

    def _loop(self):
        err_count = 0
        err_last_log = 0.0

        while not self._stop.is_set():
            try:
                frame = self._read_frame()

                if frame is not None:
                    with self._lock:
                        self._color = frame.copy()
                else:
                    time.sleep(0.05)

            except Exception as e:
                if not self._stop.is_set():
                    err_count += 1
                    now = time.monotonic()
                    if now - err_last_log > 5.0:
                        print(f"[{self.name}] capture error x{err_count}: {e}")
                        err_count = 0
                        err_last_log = now
                    time.sleep(0.1)

    def get_latest_color(self):
        with self._lock:
            return None if self._color is None else self._color.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._stop_backend()
        print(f"[{self.name}] stopped")

class OpenCVCapture(SceneCaptureBase):
    name = "OpenCVCapture"

    def __init__(
        self,
        cam_id: int = 1,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_AVFOUNDATION)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV scene camera at index {cam_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[OpenCVCapture] Camera {cam_id} started at {width}x{height}@{fps}")

        super().__init__()

    def _read_frame(self):
        self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def _stop_backend(self):
        self.cap.release()

class RealSenseCapture(SceneCaptureBase):
    name = "RealSense"

    def __init__(self, serial: str | None = None):
        import pyrealsense2 as rs

        self.rs = rs

        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError(
                "No RealSense devices found. "
                "Check the USB cable and run `rs-enumerate-devices`."
            )

        print("[RealSense] Available devices:")
        for d in devices:
            print(
                f"  - {d.get_info(rs.camera_info.name)} "
                f"(serial {d.get_info(rs.camera_info.serial_number)}, "
                f"usb {d.get_info(rs.camera_info.usb_type_descriptor)})"
            )

        if serial is not None:
            device = next(
                (
                    d for d in devices
                    if d.get_info(rs.camera_info.serial_number) == serial
                ),
                None,
            )
            if device is None:
                raise RuntimeError(f"RealSense serial {serial!r} not found")
        else:
            device = devices[0]

        chosen_serial = device.get_info(rs.camera_info.serial_number)
        usb_desc = device.get_info(rs.camera_info.usb_type_descriptor)
        fps = 30 if usb_desc.startswith("3") else 15

        print(f"[RealSense] Using serial {chosen_serial}, USB {usb_desc} → {fps} fps")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(chosen_serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        print("[RealSense] Pipeline started")

        super().__init__()

    def _read_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self.align.process(frames)
        cf = aligned.get_color_frame()

        if not cf:
            return None

        return np.asanyarray(cf.get_data()).copy()

    def _stop_backend(self):
        self.pipeline.stop()


# class RealSenseCapture:
#     """Opens one pyrealsense2 pipeline for colour (RGB) at 640×480.

#     Automatically selects USB 2.1 vs USB 3 frame rate (15 vs 30 fps).
#     A background thread continuously pulls frames into a locked slot.
#     """

#     def __init__(self, serial: str | None = None):
#         import pyrealsense2 as rs

#         devices = list(rs.context().query_devices())
#         if not devices:
#             raise RuntimeError(
#                 "No RealSense devices found. "
#                 "Check the USB cable (needs real USB-3 data cable, not charge-only) "
#                 "and run `rs-enumerate-devices`."
#             )

#         print("[RealSense] Available devices:")
#         for d in devices:
#             print(f"  - {d.get_info(rs.camera_info.name)} "
#                   f"(serial {d.get_info(rs.camera_info.serial_number)}, "
#                   f"usb {d.get_info(rs.camera_info.usb_type_descriptor)})")

#         if serial is not None:
#             match = next(
#                 (d for d in devices
#                  if d.get_info(rs.camera_info.serial_number) == serial), None
#             )
#             if match is None:
#                 raise RuntimeError(f"RealSense serial {serial!r} not found")
#             device = match
#         else:
#             device = devices[0]

#         chosen_serial = device.get_info(rs.camera_info.serial_number)
#         usb_desc      = device.get_info(rs.camera_info.usb_type_descriptor)
#         fps = 30 if usb_desc.startswith("3") else 15
#         print(f"[RealSense] Using serial {chosen_serial}, USB {usb_desc} → {fps} fps")

#         pipeline = rs.pipeline()
#         cfg = rs.config()
#         cfg.enable_device(chosen_serial)
#         cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
#         cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  fps)
#         pipeline.start(cfg)

#         self.pipeline = pipeline
#         self.align    = rs.align(rs.stream.color)
#         self._color   = None
#         self._lock    = threading.Lock()
#         self._stop    = threading.Event()
#         self._thread  = threading.Thread(target=self._loop, daemon=True)
#         self._thread.start()
#         print("[RealSense] Pipeline started")

#     def _loop(self):
#         err_count    = 0
#         err_last_log = 0.0
#         while not self._stop.is_set():
#             try:
#                 frames  = self.pipeline.wait_for_frames(timeout_ms=1000)
#                 aligned = self.align.process(frames)
#                 cf = aligned.get_color_frame()
#                 if cf:
#                     img = np.asanyarray(cf.get_data()).copy()
#                     with self._lock:
#                         self._color = img
#             except Exception as e:
#                 if not self._stop.is_set():
#                     err_count += 1
#                     now = time.monotonic()
#                     if now - err_last_log > 10.0:
#                         print(f"[RealSense] capture error x{err_count}: {e}")
#                         err_count    = 0
#                         err_last_log = now
#                     time.sleep(0.1)

#     def get_latest_color(self):
#         with self._lock:
#             return self._color

#     def release(self):
#         self._stop.set()
#         self._thread.join(timeout=2.0)
#         self.pipeline.stop()
#         print("[RealSense] Pipeline stopped")
