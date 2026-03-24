import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam

from config import CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Migrate Lambda-CBAM model to CBAMLayer model")
    p.add_argument("--old",     default="outputs/lightfakedetect_celeb.keras")
    p.add_argument("--new",     default="outputs/lightfakedetect_celeb_v2.keras")
    p.add_argument("--seq-len", type=int, default=CONFIG["max_frames"])
    p.add_argument("--img-size",type=int, default=CONFIG["img_size"])
    p.add_argument("--gru-layers", type=int, default=CONFIG["gru_layers"])
    p.add_argument("--gru-units",  type=int, default=CONFIG["gru_units"])
    return p.parse_args()
def _channel_attention_old(x, reduction_ratio=1):
    channels = x.shape[-1]
    hidden = max(1, channels // reduction_ratio)
    d1 = layers.Dense(hidden, activation="relu", use_bias=False)
    d2 = layers.Dense(channels, use_bias=False)
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    avg_out  = d2(d1(avg_pool))
    max_out  = d2(d1(max_pool))
    scale    = layers.Activation("sigmoid")(avg_out + max_out)
    scale    = layers.Reshape((1, 1, channels))(scale)
    return layers.Multiply()([x, scale])
def _spatial_attention_old(x, kernel_size=7):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x,  axis=-1, keepdims=True)
    concat   = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attn     = layers.Conv2D(1, kernel_size, padding="same",
                              activation="sigmoid", use_bias=False)(concat)
    return layers.Multiply()([x, attn])

def _cbam_old(x):
    x = _channel_attention_old(x, reduction_ratio=1)
    x = _spatial_attention_old(x, kernel_size=7)
    return x

def build_old_model(seq_len, img_size, gru_layers, gru_units):
    mobilenet = MobileNet(
        input_shape=(img_size, img_size, 3),
        include_top=False, weights=None, alpha=1.0,
    )
    inp = layers.Input(shape=(seq_len, img_size, img_size, 3), name="video_input")
    x = layers.TimeDistributed(mobilenet, name="mobilenet_td")(inp)
    x = layers.TimeDistributed(
        layers.Lambda(lambda t: _cbam_old(t), name="cbam"),
        name="cbam_td"
    )(x)
    x = layers.TimeDistributed(layers.Flatten(), name="flatten_td")(x)
    for i in range(gru_layers):
        x = layers.GRU(gru_units, return_sequences=(i < gru_layers - 1),
                       name=f"gru_{i+1}")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inp, out, name="LightFakeDetect")
class CBAMLayer(layers.Layer):
    def __init__(self, reduction_ratio=1, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]
        hidden = max(1, channels // self.reduction_ratio)
        self.ca_dense1 = layers.Dense(hidden, activation="relu", use_bias=False)
        self.ca_dense2 = layers.Dense(channels, use_bias=False)
        self.sa_conv   = layers.Conv2D(1, self.kernel_size, padding="same",
                                       activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x,  axis=[1, 2])
        avg_out  = self.ca_dense2(self.ca_dense1(avg_pool))
        max_out  = self.ca_dense2(self.ca_dense1(max_pool))
        scale    = tf.sigmoid(avg_out + max_out)
        scale    = tf.reshape(scale, (-1, 1, 1, x.shape[-1]))
        x        = x * scale
        avg_sp   = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_sp   = tf.reduce_max(x,  axis=-1, keepdims=True)
        concat   = tf.concat([avg_sp, max_sp], axis=-1)
        attn     = self.sa_conv(concat)
        return x * attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio,
                    "kernel_size":     self.kernel_size})
        return cfg


def build_new_model(seq_len, img_size, gru_layers, gru_units):
    mobilenet = MobileNet(
        input_shape=(img_size, img_size, 3),
        include_top=False, weights=None, alpha=1.0,
    )
    inp = layers.Input(shape=(seq_len, img_size, img_size, 3), name="video_input")
    x = layers.TimeDistributed(mobilenet, name="mobilenet_td")(inp)
    x = layers.TimeDistributed(
        CBAMLayer(reduction_ratio=1, name="cbam"),
        name="cbam_td"
    )(x)
    x = layers.TimeDistributed(layers.Flatten(), name="flatten_td")(x)
    for i in range(gru_layers):
        x = layers.GRU(gru_units, return_sequences=(i < gru_layers - 1),
                       name=f"gru_{i+1}")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inp, out, name="LightFakeDetect")


def main():
    args = parse_args()

    print(f"Building old architecture (seq={args.seq_len}, img={args.img_size}, "
          f"gru={args.gru_layers}×{args.gru_units})...")
    old_model = build_old_model(args.seq_len, args.img_size,
                                args.gru_layers, args.gru_units)

    print(f"Loading weights from: {args.old}")
    old_model.load_weights(args.old)

    print("Building new CBAMLayer architecture...")
    new_model = build_new_model(args.seq_len, args.img_size,
                                args.gru_layers, args.gru_units)
    dummy = tf.zeros((1, args.seq_len, args.img_size, args.img_size, 3))
    _ = new_model(dummy)
    old_by_name = {l.name: l for l in old_model.layers}
    new_by_name = {l.name: l for l in new_model.layers}

    copied, skipped = 0, 0
    for name, new_layer in new_by_name.items():
        if name in old_by_name:
            w = old_by_name[name].get_weights()
            if w:
                try:
                    new_layer.set_weights(w)
                    print(f"  ✓ {name}")
                    copied += 1
                except Exception as e:
                    print(f"  ✗ {name} — {e}")
                    skipped += 1
        else:
            skipped += 1

    print(f"\nCopied: {copied}  Skipped: {skipped}")

    new_model.compile(
        optimizer=Adam(CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    new_model.save(args.new)
    print(f"\nSaved migrated model → {args.new}")
    print(f"Use: python predict.py --model {args.new} --video <your_video.mp4>")


if __name__ == "__main__":
    main()
