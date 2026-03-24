import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam

from config import CONFIG
class CBAMLayer(layers.Layer):

    def __init__(self, reduction_ratio: int = 1, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]
        hidden = max(1, channels // self.reduction_ratio)
        self.ca_dense1 = layers.Dense(hidden, activation="relu", use_bias=False)
        self.ca_dense2 = layers.Dense(channels, use_bias=False)
        self.sa_conv = layers.Conv2D(
            1, self.kernel_size, padding="same",
            activation="sigmoid", use_bias=False
        )
        super().build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2])          
        max_pool = tf.reduce_max(x,  axis=[1, 2])         
        avg_out  = self.ca_dense2(self.ca_dense1(avg_pool))
        max_out  = self.ca_dense2(self.ca_dense1(max_pool))
        scale    = tf.sigmoid(avg_out + max_out)
        
        scale    = tf.reshape(scale, (-1, 1, 1, x.shape[-1]))
        x        = x * scale
        avg_sp = tf.reduce_mean(x, axis=-1, keepdims=True)  
        max_sp = tf.reduce_max(x,  axis=-1, keepdims=True) 
        concat = tf.concat([avg_sp, max_sp], axis=-1)      
        attn   = self.sa_conv(concat)                
        return x * attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size":     self.kernel_size,
        })
        return cfg

def build_lightfakedetect(
    seq_len:        int   = None,
    img_size:       int   = None,
    cbam_reduction: int   = None,
    gru_layers:     int   = None,
    gru_units:      int   = None,
    learning_rate:  float = None,
    freeze_backbone: bool = True,
) -> Model:
    seq_len        = seq_len        or CONFIG["max_frames"]
    img_size       = img_size       or CONFIG["img_size"]
    cbam_reduction = cbam_reduction or CONFIG["cbam_reduction_ratio"]
    gru_layers     = gru_layers     or CONFIG["gru_layers"]
    gru_units      = gru_units      or CONFIG["gru_units"]
    learning_rate  = learning_rate  or CONFIG["learning_rate"]
    mobilenet = MobileNet(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=1.0,
    )
    mobilenet.trainable = not freeze_backbone
    inp = layers.Input(
        shape=(seq_len, img_size, img_size, 3),
        name="video_input"
    )
    x = layers.TimeDistributed(mobilenet, name="mobilenet_td")(inp)
    x = layers.TimeDistributed(
        CBAMLayer(reduction_ratio=cbam_reduction, name="cbam"),
        name="cbam_td"
    )(x)
    x = layers.TimeDistributed(layers.Flatten(), name="flatten_td")(x)
    for i in range(gru_layers):
        return_seq = (i < gru_layers - 1)
        x = layers.GRU(
            gru_units,
            return_sequences=return_seq,
            name=f"gru_{i + 1}"
        )(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="LightFakeDetect")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_backbone(model: Model, learning_rate: float = 1e-5, fine_tune_at: int = 80) -> Model:
    td_layer = model.get_layer("mobilenet_td")
    mobilenet = td_layer.layer
    mobilenet.trainable = True
    if fine_tune_at > 0:
        for layer in mobilenet.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"[model] Partial unfreeze: layers {fine_tune_at}+ are trainable")
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
def build_ablation_variant(
    variant:    str   = "full",
    seq_len:    int   = None,
    img_size:   int   = None,
    gru_layers: int   = None,
    gru_units:  int   = None,
) -> Model:
en    = seq_len    or CONFIG["max_frames"]
    img_size   = img_size   or CONFIG["img_size"]
    gru_layers = gru_layers or CONFIG["gru_layers"]
    gru_units  = gru_units  or CONFIG["gru_units"]

    inp = layers.Input(shape=(seq_len, img_size, img_size, 3))
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
    if variant == "full":
        x = layers.TimeDistributed(
            CBAMLayer(reduction_ratio=CONFIG["cbam_reduction_ratio"], name="cbam"),
            name="cbam_td"
        )(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    if variant == "no_gru":
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
