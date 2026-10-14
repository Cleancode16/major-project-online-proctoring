"""
L2CS-Net eye-gaze estimator (GZ3, flow.txt §A.6 "L2CS-Net gaze").

Model: L2CS-Net, ResNet50 backbone, Gaze360-trained weights (Abdelrahman et al. 2022, MIT code; Gaze360 data is
research / non-commercial). Two 90-bin heads (4 deg bins over [-180, 176]); angle = softmax expectation.
Weights: py-feat/l2cs on Hugging Face = the official L2CSNet_gaze360.pkl converted to .safetensors (no pickle).
CPU only (user decision: CUDA not used).

Preprocessing = upstream L2CS pipeline: face-detector box crop -> RGB -> resize 224x224 -> resize to the training
input size 448 -> ImageNet normalisation.

Output convention (upstream l2cs/utils.py gaze vector (-cos p sin y, -sin p, -cos p cos y), camera x right, y down):
  first head (fc_yaw_gaze)   yaw   > 0 = gaze towards the image LEFT
  second head (fc_pitch_gaze) pitch > 0 = gaze UP
predict() returns them as h = -yaw (> 0 = towards image right) and v = -pitch (> 0 = down), the same signs the
iris-ratio method uses, so gaze.gaze_decision labels both the same way.

  CLI:  python -m video.l2cs --download     (fetch + SHA-256 check of the weights)
"""

import hashlib
import sys
import urllib.request

import cv2
import numpy as np

import config as C

WEIGHTS_URL = "https://huggingface.co/py-feat/l2cs/resolve/main/l2cs_gaze360_resnet50.safetensors"
WEIGHTS_SHA256 = "75405bdc01f7086b3887280fe50c28a14bc62463a53fa749781e4a3cf98eca0e"
DEFAULT_WEIGHTS = C.MODELS_DIR / "l2cs" / "l2cs_gaze360_resnet50.safetensors"
NUM_BINS, BIN_DEG, BIN_OFFSET = 90, 4.0, 180.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _build_model():
    import torch.nn as nn
    from torchvision.models.resnet import Bottleneck, ResNet

    class L2CS(ResNet):
        """torchvision ResNet50 (same parameter names as the checkpoint) + the two L2CS bin heads."""

        def __init__(self):
            super().__init__(Bottleneck, [3, 4, 6, 3])
            del self.fc
            self.fc_yaw_gaze = nn.Linear(2048, NUM_BINS)
            self.fc_pitch_gaze = nn.Linear(2048, NUM_BINS)
            self.fc_finetune = nn.Linear(2048 + 3, 3)       # unused at inference, kept for strict loading

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x).flatten(1)
            return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)

    return L2CS()


class L2CSGaze:
    def __init__(self, weights=DEFAULT_WEIGHTS, input_size=448, threads=None):
        import torch
        from safetensors.torch import load_file
        if threads:
            torch.set_num_threads(int(threads))
        self.torch = torch
        self.input_size = int(input_size)
        self.model = _build_model()
        self.model.load_state_dict(load_file(str(weights)), strict=True)
        self.model.eval()
        self.centres = torch.arange(NUM_BINS, dtype=torch.float32) * BIN_DEG - BIN_OFFSET
        blank = np.full((self.input_size, self.input_size, 3), 128, np.uint8)
        for _ in range(2):          # warm-up: the first CPU passes are ~10x slower (kernel selection)
            self.predict(blank, (0, 0, self.input_size, self.input_size))

    def crop(self, frame_bgr, bbox, margin=0.0):
        """Face crop from an (x, y, w, h) detector box, optionally enlarged by `margin` per side."""
        H, W = frame_bgr.shape[:2]
        x, y, w, h = bbox
        x0, y0 = int(max(0, x - margin * w)), int(max(0, y - margin * h))
        x1, y1 = int(min(W, x + w + margin * w)), int(min(H, y + h + margin * h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return frame_bgr[y0:y1, x0:x1]

    def predict(self, frame_bgr, bbox, margin=0.0):
        """Returns {"yaw_deg", "pitch_deg", "h", "v"} for the face in `bbox`, or None if the crop is unusable."""
        face_img = self.crop(frame_bgr, bbox, margin)
        if face_img is None:
            return None
        img = cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), (224, 224))
        if self.input_size != 224:
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = self.torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
        with self.torch.inference_mode():
            yaw_logits, pitch_logits = self.model(x)
            yaw = float((self.torch.softmax(yaw_logits, 1)[0] * self.centres).sum())
            pitch = float((self.torch.softmax(pitch_logits, 1)[0] * self.centres).sum())
        return {"yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2), "h": -yaw, "v": -pitch}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(dest=DEFAULT_WEIGHTS):
    """Download the weights (if missing) and verify their SHA-256. Returns the path."""
    if dest.exists() and sha256(dest) == WEIGHTS_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading L2CS-Net Gaze360 weights (~96 MB) from {WEIGHTS_URL}")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    got = sha256(tmp)
    if got != WEIGHTS_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"L2CS weights checksum mismatch: {got}")
    tmp.replace(dest)
    return dest


if __name__ == "__main__":
    if "--download" in sys.argv:
        print(f"L2CS weights ready: {download()}")
    else:
        print(__doc__)
