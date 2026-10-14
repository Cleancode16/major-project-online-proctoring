"""
Face anti-spoofing with MiniFASNet (FAS1, flow.txt §A.7b "Face anti-spoofing").

Model: Silent-Face-Anti-Spoofing (minivision-ai, Apache-2.0): MiniFASNetV2 (context scale 2.7) + MiniFASNetV1SE
(scale 4.0), 80x80 inputs, 3-class softmax, class 1 = real face. The two models' softmax outputs are summed and
real score = sum[1] / 2 (upstream test.py). CPU only.
Weights: the repository's resources/anti_spoof_models/*.pth (git blob SHA-1 checked), loaded once with
torch.load(weights_only=True) and stored as .safetensors in server/models/antispoof/ (no pickle at run time).

Preprocessing = upstream CropImage + ToTensor: square-ish crop of `scale` x the face box around its centre,
shifted inside the frame, resized to 80x80, BGR, float WITHOUT /255 (upstream functional.to_tensor).

  CLI:  python -m video.antispoof --download     (fetch the .pth files, check SHA-1, convert to .safetensors)
"""

import hashlib
import sys
import urllib.request

import cv2
import numpy as np

import config as C

MODELS = [  # (file stem, architecture, crop scale)
    ("2.7_80x80_MiniFASNetV2", "MiniFASNetV2", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE", "MiniFASNetV1SE", 4.0),
]
REPO = "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/"
GIT_SHA1 = {"2.7_80x80_MiniFASNetV2": "47c4af2023fb072c0f5b0e0ade4824053a0558e1",
            "4_0_0_80x80_MiniFASNetV1SE": "55a25b316ef33ced3925687007a31a5e306990db"}
DEFAULT_DIR = C.MODELS_DIR / "antispoof"
INPUT = 80
KERNEL = ((INPUT + 15) // 16, (INPUT + 15) // 16)       # upstream get_kernel(80, 80) = (5, 5)


def crop_box(src_w, src_h, bbox, scale):
    """Upstream CropImage._get_new_box: scale x box around its centre, clamped / shifted into the frame."""
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    x1, y1, x2, y2 = cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > src_w - 1:
        x1 -= x2 - src_w + 1
        x2 = src_w - 1
    if y2 > src_h - 1:
        y1 -= y2 - src_h + 1
        y2 = src_h - 1
    return int(x1), int(y1), int(x2), int(y2)


class AntiSpoof:
    def __init__(self, models_dir=DEFAULT_DIR):
        import torch
        from safetensors.torch import load_file

        from video import minifasnet
        self.torch = torch
        self.models = []
        for stem, arch, scale in MODELS:
            m = getattr(minifasnet, arch)(conv6_kernel=KERNEL)
            m.load_state_dict(load_file(str(models_dir / f"{stem}.safetensors")), strict=True)
            m.eval()
            self.models.append((m, scale))

    def real_score(self, frame_bgr, bbox):
        """Probability-like real-face score in [0, 1] for the face in `bbox` (x, y, w, h); None if unusable."""
        H, W = frame_bgr.shape[:2]
        if bbox[2] < 8 or bbox[3] < 8:
            return None
        pred = np.zeros(3, np.float64)
        with self.torch.inference_mode():
            for model, scale in self.models:
                x1, y1, x2, y2 = crop_box(W, H, bbox, scale)
                patch = cv2.resize(frame_bgr[y1:y2 + 1, x1:x2 + 1], (INPUT, INPUT))
                x = self.torch.from_numpy(patch.transpose(2, 0, 1).astype(np.float32))[None]
                pred += self.torch.softmax(model(x), dim=1)[0].numpy()
        return float(pred[1] / len(self.models))


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(models_dir=DEFAULT_DIR):
    """Fetch the upstream .pth weights, verify their git SHA-1, convert to .safetensors (weights_only load)."""
    import io

    import torch
    from safetensors.torch import save_file
    models_dir.mkdir(parents=True, exist_ok=True)
    for stem, _, _ in MODELS:
        dest = models_dir / f"{stem}.safetensors"
        if dest.exists():
            continue
        print(f"Downloading {stem}.pth")
        data = urllib.request.urlopen(REPO + f"{stem}.pth", timeout=60).read()
        if git_blob_sha1(data) != GIT_SHA1[stem]:
            raise RuntimeError(f"{stem}: checksum mismatch")
        sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        sd = {k[7:] if k.startswith("module.") else k: v.contiguous() for k, v in sd.items()}
        save_file(sd, str(dest))
    return models_dir


if __name__ == "__main__":
    if "--download" in sys.argv:
        print(f"Anti-spoofing weights ready in {download()}")
    else:
        print(__doc__)
