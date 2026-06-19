import base64
import io
from typing import Any

import numpy as np
import requests
from PIL import Image


def pil_to_b64_jpeg(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class RemoteMolmoActPolicy:
    """
    Drop-in-ish replacement for MolmoActPolicy.

    It exposes the same method used by runtime.py:

        predict_chunk(images, state, prompt, num_steps, cuda_graph)

    but forwards the request to a remote GPU server.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 20.0,
        jpeg_quality: int = 85,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

    def predict_chunk(
        self,
        images,
        state: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 10,
        cuda_graph: bool = False,
    ) -> np.ndarray:
        """
        Args:
            images: [scene, wrist] PIL images
            state: model-frame joint state
            prompt: task string

        Returns:
            (T, JOINT_COUNT) action chunk in model frame
        """
        if len(images) != 2:
            raise ValueError(f"Expected images=[scene, wrist], got {len(images)} images")

        scene, wrist = images

        payload = {
            "prompt": prompt,
            "scene_image_b64": pil_to_b64_jpeg(scene, self.jpeg_quality),
            "wrist_image_b64": pil_to_b64_jpeg(wrist, self.jpeg_quality),
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "num_steps": int(num_steps),
            "cuda_graph": bool(cuda_graph),
        }

        response = requests.post(
            f"{self.url}/predict_chunk",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        actions = np.asarray(data["actions"], dtype=np.float32)

        if actions.ndim != 2:
            raise RuntimeError(f"Expected 2-D action chunk, got shape {actions.shape}")

        return actions