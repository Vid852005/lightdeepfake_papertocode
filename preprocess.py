# =============================================================================
# preprocess.py — MTCNN face detection + SSIM deduplication pipeline
#
# Changes from original, all documented:
#
# [Paper-faithful]
#   - Padding uses frame repetition (not np.zeros) — paper Section 3.3:
#     "duplicating frames as needed". Zero-padding introduced black frames
#     that the GRU learned as an artefact signal.
#
# [Reasoned extension — paper Section 6 / Figure 6]
#   - augment_frames(): horizontal flip + brightness jitter on training crops.
#     The paper's own misclassification analysis (Figure 6) shows illumination
#     changes and facial occlusions cause false positives. Brightness jitter
#     directly targets the illumination failure mode.
#     Only applied during build_dataset(is_train=True).
#
# [Engineering fix]
#   - Memory-safe streaming: build_dataset() yields per-video and stacks only
#     at the end. Previously the entire dataset was held twice in RAM during
#     np.array(X) conversion.
#   - PyTorch/MTCNN is CPU-only; torch.set_num_threads set to match CONFIG.
# =============================================================================

import os
import random

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from config import CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────────────────────

# Force CPU — no CUDA on AMD Radeon iGPU without ROCm
DEVICE = "cpu"
torch.set_num_threads(CONFIG["tf_inter_op_threads"])

_mtcnn = MTCNN(
    image_size=CONFIG["img_size"],
    margin=20,
    min_face_size=20,
    thresholds=[0.6, 0.7, CONFIG["mtcnn_conf_high"]],
    factor=0.709,
    post_process=False,
    device=DEVICE,
    keep_all=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Face detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_crop_face(
    frame_bgr: np.ndarray,
    conf_threshold: float = 0.99
) -> np.ndarray | None:
    """
    Detect the largest face in a BGR frame using MTCNN.

    Returns img_size × img_size RGB array, or None if no face passes threshold.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        boxes, probs = _mtcnn.detect(frame_rgb)
        if boxes is None or len(boxes) == 0:
            return None

        valid = [
            (b, p) for b, p in zip(boxes, probs)
            if p is not None and p >= conf_threshold
        ]
        if not valid:
            return None

        # Largest bounding box
        box = max(valid, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))[0]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame_rgb.shape[1], x2)
        y2 = min(frame_rgb.shape[0], y2)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (CONFIG["img_size"], CONFIG["img_size"]))

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SSIM deduplication
# ─────────────────────────────────────────────────────────────────────────────

def filter_frames_ssim(
    frames: list[np.ndarray],
    threshold: float = 0.85
) -> list[np.ndarray]:
    """
    Remove near-duplicate consecutive frames using SSIM.

    Paper Section 3.3: initial threshold=0.85, adaptive up to 0.97.
    """
    if not frames:
        return frames

    unique = [frames[0]]
    for frame in frames[1:]:
        prev_gray = cv2.cvtColor(unique[-1], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frame,     cv2.COLOR_RGB2GRAY)
        score = ssim(prev_gray, curr_gray, data_range=255)
        if score < threshold:
            unique.append(frame)

    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (extension — motivated by paper Section 6 / Figure 6)
# ─────────────────────────────────────────────────────────────────────────────

def augment_frames(frames: np.ndarray, label: int) -> np.ndarray:
    """
    Apply training-time augmentation to a [T, H, W, 3] float32 array.

    - Horizontal flip (50% chance, all labels):
        Deepfake artifacts are symmetric; flipping preserves the detection signal.
    - Brightness jitter ±10% (50% chance, all labels):
        Targets illumination-induced misclassification (paper Figure 6).
        Real videos misclassified as fake in 19/21 cases, largely due to
        lighting changes per paper Section 6.

    Parameters
    ----------
    frames : float32 array in [0, 1]
    label  : class label (kept as parameter for possible label-conditional logic)
    """
    if random.random() > 0.5:
        frames = frames[:, :, ::-1, :].copy()  # horizontal flip

    if random.random() > 0.5:
        delta = random.uniform(-0.10, 0.10)
        frames = np.clip(frames + delta, 0.0, 1.0)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Per-video preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_video(
    video_path:     str,
    label:          int,
    conf_threshold: float | None = None,
) -> tuple[np.ndarray, int] | None:
    """
    Full preprocessing pipeline for one video (paper Section 3.3):
      1. Frame extraction (sub-sample to ≤100 for long videos).
      2. MTCNN face detection & crop.
      3. SSIM deduplication with adaptive threshold.
      4. Pad / trim to [min_frames, max_frames].

    Padding uses frame repetition (paper: "duplicating frames as needed").
    Previous implementation used np.zeros which created artifactual black frames.

    Returns (frames_array [T, H, W, 3] float32 in [0,1], label) or None.
    """
    if conf_threshold is None:
        conf_threshold = CONFIG["mtcnn_conf_high"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sub-sample long videos (paper DFDC Section 3.3)
    if total_frames > 100:
        indices = sorted(random.sample(range(total_frames), 100))
    else:
        indices = list(range(total_frames))

    raw_frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        crop = detect_and_crop_face(frame, conf_threshold)
        if crop is not None:
            raw_frames.append(crop)
    cap.release()

    # Retry with relaxed confidence if insufficient faces
    if len(raw_frames) < CONFIG["min_frames"] and conf_threshold > CONFIG["mtcnn_conf_low"]:
        return preprocess_video(video_path, label, CONFIG["mtcnn_conf_low"])

    # Adaptive SSIM deduplication
    threshold = CONFIG["ssim_threshold_init"]
    unique_frames = filter_frames_ssim(raw_frames, threshold)

    while len(unique_frames) < CONFIG["min_frames"] and threshold < CONFIG["ssim_threshold_max"]:
        threshold += 0.02
        unique_frames = filter_frames_ssim(raw_frames, threshold)

    # Frame repetition padding (paper Section 3.3)
    # Fix: previous code used np.zeros (black frames) — replaced with repeat
    if len(unique_frames) < CONFIG["min_frames"]:
        while len(unique_frames) < CONFIG["min_frames"]:
            unique_frames.append(random.choice(unique_frames))

    # Trim to max_frames
    if len(unique_frames) > CONFIG["max_frames"]:
        unique_frames = random.sample(unique_frames, CONFIG["max_frames"])

    frames_array = np.array(unique_frames, dtype=np.float32) / 255.0
    return frames_array, label


# ─────────────────────────────────────────────────────────────────────────────
# Dataset path loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_video_paths(
    real_dir:   str,
    fake_dir:   str,
    max_videos: int | None = None,
) -> list[tuple[str, int]]:
    """
    Collect (video_path, label) pairs.
    If max_videos is set, performs BALANCED sampling (equal Real/Fake).
    """
    exts = (".mp4", ".avi", ".mov")
    real_samples: list[tuple[str, int]] = []
    fake_samples: list[tuple[str, int]] = []

    if os.path.exists(real_dir):
        for fname in os.listdir(real_dir):
            if fname.lower().endswith(exts):
                real_samples.append((os.path.join(real_dir, fname), 0))

    if os.path.exists(fake_dir):
        for fname in os.listdir(fake_dir):
            if fname.lower().endswith(exts):
                fake_samples.append((os.path.join(fake_dir, fname), 1))

    if max_videos:
        n_per_class = max_videos // 2
        # Sample real
        if len(real_samples) > n_per_class:
            real_samples = random.sample(real_samples, n_per_class)
        # Sample fake
        if len(fake_samples) > n_per_class:
            fake_samples = random.sample(fake_samples, n_per_class)
        
        print(f"[preprocess] Balanced sampling: {len(real_samples)} real, {len(fake_samples)} fake")

    samples = real_samples + fake_samples
    random.shuffle(samples)
    return samples


def load_dfdc_paths(
    dfdc_dir:   str,
    max_videos: int = 5372,
) -> list[tuple[str, int]]:
    """Load (video_path, label) pairs from DFDC using metadata.json."""
    import json
    meta_path = os.path.join(dfdc_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    samples: list[tuple[str, int]] = []
    for fname, info in meta.items():
        path = os.path.join(dfdc_dir, fname)
        if not os.path.exists(path):
            continue
        label = 1 if info["label"] == "FAKE" else 0
        samples.append((path, label))

    if len(samples) > max_videos:
        samples = random.sample(samples, max_videos)

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    samples:        list[tuple[str, int]],
    desc:           str,
    checkpoint_dir: str,
    is_train:       bool = False,
    save_only:      bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a list of videos and save/return as numpy arrays.
    
    Parameters
    ----------
    save_only : bool
        If True, only saves .npy checkpoints and returns empty arrays.
        Crucial for 16GB RAM stability with large datasets.
    """
    T = CONFIG["max_frames"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    X: list[np.ndarray] = []
    y: list[int]        = []
    skipped = 0
    resumed = 0

    # Memory Safety: Use streaming mode if explicit or if dataset is large (>100 per paper ref)
    streaming_mode = save_only or len(samples) > 100
    if streaming_mode:
        print(f"[preprocess] Streaming mode enabled for {len(samples)} videos (memory safety)")

    for path, label in tqdm(samples, desc=desc):
        video_id  = os.path.splitext(os.path.basename(path))[0]
        ckpt_file = os.path.join(checkpoint_dir, f"{video_id}_{label}.npy")

        # ── Load from checkpoint ──────────────────────────────────────────
        if os.path.exists(ckpt_file):
            if not streaming_mode:
                frames = np.load(ckpt_file)
                if is_train:
                    frames = augment_frames(frames, label)
                X.append(frames)
                y.append(label)
            resumed += 1
            continue

        # ── Process video ─────────────────────────────────────────────────
        result = preprocess_video(path, label)
        if result is None:
            skipped += 1
            continue

        frames, lbl = result

        # Pad (frame repetition) or trim to fixed length T
        if len(frames) < T:
            while len(frames) < T:
                frames = np.concatenate(
                    [frames, frames[:T - len(frames)]], axis=0
                )
        frames = frames[:T]

        # Save checkpoint (raw, pre-augmentation — augment at load time)
        np.save(ckpt_file, frames)

        if not streaming_mode:
            if is_train:
                frames = augment_frames(frames, lbl)
            X.append(frames)
            y.append(lbl)

    print(f"\n  Resumed from checkpoint : {resumed}")
    print(f"  Newly processed         : {len(samples) - resumed - skipped}")
    print(f"  Skipped (no face found) : {skipped}")

    if streaming_mode:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
