import base64
import io
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from molmoact_so101.model.policy import MolmoActPolicy, REPO_ID


app = FastAPI()
policy: MolmoActPolicy | None = None


class PredictRequest(BaseModel):
    prompt: str
    scene_image_b64: str
    wrist_image_b64: str
    state: List[float]
    num_steps: int = 10
    cuda_graph: bool = False


class PredictResponse(BaseModel):
    actions: List[List[float]]


def decode_image_b64(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


@app.on_event("startup")
def startup():
    global policy

    print("[remote] Loading MolmoActPolicy...")
    policy = MolmoActPolicy.from_pretrained(
        REPO_ID,
        dtype="bfloat16",
        device="cuda",
        apply_patches=False,
    )
    print("[remote] Policy ready.")


@app.get("/health")
def health():
    return {"status": "ok", "policy_loaded": policy is not None}


@app.post("/predict_chunk", response_model=PredictResponse)
def predict_chunk(req: PredictRequest):
    assert policy is not None, "Policy is not loaded"

    scene = decode_image_b64(req.scene_image_b64)
    wrist = decode_image_b64(req.wrist_image_b64)

    state = np.asarray(req.state, dtype=np.float32)

    with torch.inference_mode():
        actions = policy.predict_chunk(
            images=[scene, wrist],   # important: [scene, wrist]
            state=state,
            prompt=req.prompt,
            num_steps=req.num_steps,
            cuda_graph=req.cuda_graph,
        )

    actions = np.asarray(actions, dtype=np.float32)
    return PredictResponse(actions=actions.tolist())