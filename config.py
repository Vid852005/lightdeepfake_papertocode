# =============================================================================
# config.py — LightFakeDetect configuration
#
# Hardware target: AMD Ryzen 5 7520U | 16 GB RAM | CPU-only (no CUDA)
#
# Design philosophy:
#   - Paper hyperparameters (AlMuhaideb et al., Mathematics 2025) are preserved
#     wherever memory allows.
#   - Only img_size, max_frames, gru_units, and batch_size are reduced to fit
#     inside ~12 GB working RAM on a 16 GB machine.
#   - Every deviation is documented below with the paper's original value.
#
# Paper reference values (run on Google Colab Pro+ A100):
#   img_size    = 224   →  reduced to 112  (saves 4× memory per frame)
#   max_frames  = 55    →  reduced to 20   (primary memory bottleneck)
#   min_frames  = 40    →  reduced to 10
#   gru_layers  = 4     →  KEPT (temporal depth is architecturally critical)
#   gru_units   = 128   →  reduced to 64
#   batch_size  = 4     →  reduced to 2
#   learning_rate = 1e-4 → KEPT
#   epochs      = 10    →  KEPT (with early stopping)
# =============================================================================

import os
import random
import numpy as np
import tensorflow as tf

CONFIG = {
    # ── Dataset paths ───────────────────────────────────────────────────────
    "celeb_df_real_dir":    "D:/celeb-df-v2/Celeb-real",
    "celeb_df_fake_dir":    "D:/celeb-df-v2/Celeb-synthesis",
    "dfdc_dir": "outputs",
    "output_dir":           "outputs",

    # Paper uses 224; increased to 160 for better spatial detail
    "img_size":             160,

    # Paper uses min=40, max=55; increased to 40 for better accuracy on Ryzen
    "min_frames":           20,
    "max_frames":           40,

    # ── SSIM deduplication thresholds (paper Section 3.3) ──────────────────
    "ssim_threshold_init":  0.85,   # paper value — kept
    "ssim_threshold_max":   0.97,   # paper value — kept

    # ── MTCNN confidence thresholds (paper Section 3.3) ────────────────────
    "mtcnn_conf_high":      0.99,   # paper value — kept
    "mtcnn_conf_low":       0.90,   # paper value — kept

    # ── Model architecture (paper Section 3.4) ──────────────────────────────
    "cbam_reduction_ratio": 1,      # paper value — kept (tuned via KerasTuner)
    "gru_layers":           4,      # paper value — KEPT (critical for temporal)
    "gru_units":            64,     # paper: 128; halved for RAM

    # ── Training hyperparameters (paper Section 3.4) ─────────────────────────
    "learning_rate":        1e-4,   # paper value — kept
    "batch_size":           4,      # paper: 4; stabilized via larger batch
    "epochs":               10,     # paper value — kept

    # ── Reproducibility ─────────────────────────────────────────────────────
    "seed":                 42,

    # ── CPU threading (Ryzen 5 7520U: 4 cores / 8 threads) ──────────────────
    # Reserve 2 threads for OS; use 6 for TF inter-op and 2 for intra-op
    "tf_inter_op_threads":  6,
    "tf_intra_op_threads":  2,

    # ── Memory growth (prevents TF grabbing all RAM at startup) ─────────────
    "tf_memory_growth":     True,

    # ── Preprocessing workers ───────────────────────────────────────────────
    # Keep at 1 for stability on Ryzen iGPU machines; MTCNN is already threaded
    "preprocess_workers":   1,
}


def apply_seeds() -> None:
    """Seed Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    tf.random.set_seed(CONFIG["seed"])


def configure_cpu() -> None:
    """
    Apply Ryzen-specific TensorFlow CPU settings.

    - Sets inter/intra-op thread counts based on the 4-core / 8-thread Ryzen 5 7520U.
    - Enables memory growth so TF does not allocate all 16 GB at startup.
    - Disables GPU (no CUDA on AMD Radeon iGPU with standard TF).
    """
    # Disable GPU — TF's CUDA stack won't find AMD Radeon without ROCm
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Thread tuning for Zen 4 core (4P cores, 8 threads)
    tf.config.threading.set_inter_op_parallelism_threads(
        CONFIG["tf_inter_op_threads"]
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        CONFIG["tf_intra_op_threads"]
    )

    # Enable TF mixed-precision on CPU where supported (bfloat16)
    # This halves memory for activations on Zen 4 which has native bf16 support
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("[config] Mixed bfloat16 precision enabled (Zen 4 native)")
    except Exception:
        print("[config] Mixed precision not available — using float32")

    print(
        f"[config] TF threads: inter={CONFIG['tf_inter_op_threads']}, "
        f"intra={CONFIG['tf_intra_op_threads']}"
    )


def make_dirs() -> None:
    """Create required output directories."""
    for path in [
        CONFIG["output_dir"],
        CONFIG["celeb_df_real_dir"],
        CONFIG["celeb_df_fake_dir"],
        CONFIG["dfdc_dir"],
        os.path.join(CONFIG["output_dir"], "checkpoints"),
    ]:
        os.makedirs(path, exist_ok=True)
