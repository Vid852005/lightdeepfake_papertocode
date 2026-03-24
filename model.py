# =============================================================================
# model.py — LightFakeDetect architecture
#
# Architecture (paper Section 3.1):
#   Video frames → [TimeDistributed] MobileNet V1
#                → [TimeDistributed] CBAM
#                → [TimeDistributed] Flatten
#                → Stacked GRU layers (paper: 4 layers × 128 units)
#                → Dense(1) + Sigmoid
#
# Two-phase training support (extension beyond paper):
#   Phase 1 — MobileNet frozen, new heads warm up       (epochs 0..WARMUP-1)
#   Phase 2 — MobileNet unfrozen, full fine-tune at 1e-5 (epochs WARMUP..end)
#   Motivation: prevents ImageNet weights being destroyed before GRU/CBAM
#   converge. Especially important on CPU where convergence is slower.
#   Reference: Howard & Ruder (2018) "Universal Language Model Fine-tuning"
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam

from config import CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# CBAM as a proper Keras Layer subclass
# (Lambda-based CBAM is blocked in Keras 2.15 when it contains trainable vars)
# ─────────────────────────────────────────────────────────────────────────────

class CBAMLayer(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM).

    Applies channel attention then spatial attention sequentially,
    as described in Woo et al. (ECCV 2018) and used in paper Section 3.1.2.

    Parameters
    ----------
    reduction_ratio : int
        Channel compression ratio for the shared MLP.
        Paper's KerasTuner found ratio=1 optimal (Section 3.4).
    kernel_size : int
        Conv kernel for spatial attention. Paper uses 7 (standard CBAM value).
    """

    def __init__(self, reduction_ratio: int = 1, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]
        hidden = max(1, channels // self.reduction_ratio)

        # Shared MLP for channel attention
        self.ca_dense1 = layers.Dense(hidden, activation="relu", use_bias=False)
        self.ca_dense2 = layers.Dense(channels, use_bias=False)

        # Single conv for spatial attention
        self.sa_conv = layers.Conv2D(
            1, self.kernel_size, padding="same",
            activation="sigmoid", use_bias=False
        )
        super().build(input_shape)

    def call(self, x):
        # ── Channel attention (paper Eq. 2) ──
        avg_pool = tf.reduce_mean(x, axis=[1, 2])           # [B, C]
        max_pool = tf.reduce_max(x,  axis=[1, 2])           # [B, C]
        avg_out  = self.ca_dense2(self.ca_dense1(avg_pool))
        max_out  = self.ca_dense2(self.ca_dense1(max_pool))
        scale    = tf.sigmoid(avg_out + max_out)
        scale    = tf.reshape(scale, (-1, 1, 1, x.shape[-1]))
        x        = x * scale

        # ── Spatial attention (paper Eq. 3) ──
        avg_sp = tf.reduce_mean(x, axis=-1, keepdims=True)  # [B, H, W, 1]
        max_sp = tf.reduce_max(x,  axis=-1, keepdims=True)  # [B, H, W, 1]
        concat = tf.concat([avg_sp, max_sp], axis=-1)       # [B, H, W, 2]
        attn   = self.sa_conv(concat)                       # [B, H, W, 1]
        return x * attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size":     self.kernel_size,
        })
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Primary model
# ─────────────────────────────────────────────────────────────────────────────

def build_lightfakedetect(
    seq_len:        int   = None,
    img_size:       int   = None,
    cbam_reduction: int   = None,
    gru_layers:     int   = None,
    gru_units:      int   = None,
    learning_rate:  float = None,
    freeze_backbone: bool = True,
) -> Model:
    """
    Build the LightFakeDetect model.

    All parameters default to CONFIG values so callers can override selectively.

    Parameters
    ----------
    freeze_backbone : bool
        If True, MobileNet weights are frozen (Phase 1 warm-up).
        Set to False for Phase 2 fine-tuning. Default True.

    Returns
    -------
    Compiled Keras Model.
    """
    # Resolve defaults from CONFIG
    seq_len        = seq_len        or CONFIG["max_frames"]
    img_size       = img_size       or CONFIG["img_size"]
    cbam_reduction = cbam_reduction or CONFIG["cbam_reduction_ratio"]
    gru_layers     = gru_layers     or CONFIG["gru_layers"]
    gru_units      = gru_units      or CONFIG["gru_units"]
    learning_rate  = learning_rate  or CONFIG["learning_rate"]

    # ── MobileNet backbone ──────────────────────────────────────────────────
    # Paper Section 3.1.1: MobileNet V1, alpha=1, no top, ImageNet weights.
    # img_size=112 → output feature map: 3×3×1024
    # img_size=224 → output feature map: 7×7×1024  (paper config)
    mobilenet = MobileNet(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=1.0,
    )
    mobilenet.trainable = not freeze_backbone

    # ── Model graph ─────────────────────────────────────────────────────────
    inp = layers.Input(
        shape=(seq_len, img_size, img_size, 3),
        name="video_input"
    )

    # Feature extraction per frame
    x = layers.TimeDistributed(mobilenet, name="mobilenet_td")(inp)

    # Attention refinement per frame (paper Section 3.1.2)
    x = layers.TimeDistributed(
        CBAMLayer(reduction_ratio=cbam_reduction, name="cbam"),
        name="cbam_td"
    )(x)

    # Flatten spatial dims before GRU
    x = layers.TimeDistributed(layers.Flatten(), name="flatten_td")(x)

    # Temporal modeling (paper Section 3.1.3)
    # Paper: 4 layers × 128 units. This impl uses CONFIG values.
    for i in range(gru_layers):
        return_seq = (i < gru_layers - 1)
        x = layers.GRU(
            gru_units,
            return_sequences=return_seq,
            name=f"gru_{i + 1}"
        )(x)

    # Classification head
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="LightFakeDetect")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_backbone(model: Model, learning_rate: float = 1e-5, fine_tune_at: int = 80) -> Model:
    """
    Phase 2: unfreeze MobileNet for full or partial fine-tuning.

    Parameters
    ----------
    fine_tune_at : int
        The layer index from which to start unfreezing.
        MobileNetV1 (alpha=1.0) has ~86 layers (including activations/padding).
        Default 80 unfreezes only the last few layers (GlobalAvgPool and just above).
        Set to 0 for full unfreeze (original behavior).
    """
    td_layer = model.get_layer("mobilenet_td")
    mobilenet = td_layer.layer
    mobilenet.trainable = True

    # If partial unfreeze requested
    if fine_tune_at > 0:
        for layer in mobilenet.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"[model] Partial unfreeze: layers {fine_tune_at}+ are trainable")

    # Recompile after changing trainable state
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    trainable_count = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"[model] Backbone updated — trainable params: {trainable_count:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Small CNN backbone for ablation (replaces MobileNet)
# ─────────────────────────────────────────────────────────────────────────────

def _small_cnn(input_shape: tuple) -> Model:
    """Lightweight 3-layer CNN used in the standard_cnn ablation variant."""
    i = layers.Input(shape=input_shape)
    c = layers.Conv2D(32,  3, activation="relu", padding="same")(i)
    c = layers.MaxPooling2D()(c)
    c = layers.Conv2D(64,  3, activation="relu", padding="same")(c)
    c = layers.MaxPooling2D()(c)
    c = layers.Conv2D(128, 3, activation="relu", padding="same")(c)
    c = layers.GlobalAveragePooling2D()(c)
    return Model(i, c, name="small_cnn")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variants
# ─────────────────────────────────────────────────────────────────────────────

def build_ablation_variant(
    variant:    str   = "full",
    seq_len:    int   = None,
    img_size:   int   = None,
    gru_layers: int   = None,
    gru_units:  int   = None,
) -> Model:
    """
    Build one of four ablation variants (paper Table 3 / Section 5):

    - "full"         : MobileNet + CBAM + GRU  (= primary model)
    - "no_cbam"      : MobileNet + GRU, no attention
    - "no_gru"       : MobileNet + CBAM + Dense, no temporal
    - "standard_cnn" : Small CNN + GRU, no MobileNet

    All variants use CONFIG values for gru_layers / gru_units so the ablation
    is always consistent with whatever config the main model was trained under.
    Fix: previously these were hardcoded, making ablation results incomparable
    to the main model when running under reduced config.
    """
    seq_len    = seq_len    or CONFIG["max_frames"]
    img_size   = img_size   or CONFIG["img_size"]
    gru_layers = gru_layers or CONFIG["gru_layers"]
    gru_units  = gru_units  or CONFIG["gru_units"]

    inp = layers.Input(shape=(seq_len, img_size, img_size, 3))

    # ── Feature extractor ──────────────────────────────────────────────────
    if variant == "standard_cnn":
        backbone = _small_cnn((img_size, img_size, 3))
        x = layers.TimeDistributed(backbone)(inp)
    else:
        mobilenet = MobileNet(
            input_shape=(img_size, img_size, 3),
            include_top=False, weights="imagenet", alpha=1.0,
        )
        mobilenet.trainable = True
        x = layers.TimeDistributed(mobilenet)(inp)

    # ── Attention (only for "full" variant) ────────────────────────────────
    if variant == "full":
        x = layers.TimeDistributed(
            CBAMLayer(reduction_ratio=CONFIG["cbam_reduction_ratio"], name="cbam"),
            name="cbam_td"
        )(x)

    # ── Temporal module ────────────────────────────────────────────────────
    x = layers.TimeDistributed(layers.Flatten())(x)

    if variant == "no_gru":
        # Replace GRU with flat Dense to isolate temporal contribution
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
    else:
        for i in range(gru_layers):
            x = layers.GRU(gru_units, return_sequences=(i < gru_layers - 1))(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(
        inputs=inp, outputs=out,
        name=f"LightFakeDetect_{variant}"
    )
    model.compile(
        optimizer=Adam(CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
