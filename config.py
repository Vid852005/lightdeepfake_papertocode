import os
import random
import numpy as np
import tensorflow as tf

CONFIG = {
    "celeb_df_real_dir":    "D:/celeb-df-v2/Celeb-real",
    "celeb_df_fake_dir":    "D:/celeb-df-v2/Celeb-synthesis",
    "dfdc_dir": "outputs",
    "output_dir":           "outputs",
    "img_size":             160,
    "min_frames":           20,
    "max_frames":           40,
    "ssim_threshold_init":  0.85,
    "ssim_threshold_max":   0.97, 
    "mtcnn_conf_high":      0.99,  
    "mtcnn_conf_low":       0.90,  
    "cbam_reduction_ratio": 1,  
    "gru_layers":           4,     
    "gru_units":            64,  
    "learning_rate":        1e-4, 
    "batch_size":           4,      
    "epochs":               10,     
    "seed":                 42,
    "tf_inter_op_threads":  6,
    "tf_intra_op_threads":  2,
    "tf_memory_growth":     True,
    "preprocess_workers":   1,
}

def apply_seeds() -> None:
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    tf.random.set_seed(CONFIG["seed"])
def configure_cpu() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.threading.set_inter_op_parallelism_threads(
        CONFIG["tf_inter_op_threads"]
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        CONFIG["tf_intra_op_threads"]
    )
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
    for path in [
        CONFIG["output_dir"],
        CONFIG["celeb_df_real_dir"],
        CONFIG["celeb_df_fake_dir"],
        CONFIG["dfdc_dir"],
        os.path.join(CONFIG["output_dir"], "checkpoints"),
    ]:
        os.makedirs(path, exist_ok=True)
