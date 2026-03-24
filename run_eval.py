import os
import numpy as np
import tensorflow as tf
from model import CBAMLayer
from evaluate import evaluate_model
from config import configure_cpu

configure_cpu()

model = tf.keras.models.load_model(
    'outputs/lightfakedetect_celeb.keras',
    custom_objects={'CBAMLayer': CBAMLayer}
)

ckpt_dir = 'outputs/checkpoints/celeb'
test_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.npy')][:60]
X = np.array([np.load(os.path.join(ckpt_dir, f)) for f in test_files], dtype=np.float32)
y = np.array([int(f.split('_')[-1].replace('.npy', '')) for f in test_files], dtype=np.int32)

evaluate_model(model, X, y, 'Celeb-DF_v2')