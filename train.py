# =============================================================================
# train.py — Optimized LightFakeDetect (BIAS-FIXED VERSION)
# =============================================================================

import argparse
import os
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from config import CONFIG, apply_seeds, configure_cpu, make_dirs
from evaluate import evaluate_model, plot_training_curves
from model import build_lightfakedetect, unfreeze_backbone
from preprocess import build_dataset, load_video_paths, load_dfdc_paths

WARMUP_EPOCHS = 3

# =============================================================================
# Interleaved Balanced Batch Generator
# =============================================================================

class NpySequence(Sequence):
    """
    Strictly Interleaved Generator: Ensures every batch is 50/50 Real/Fake.
    This prevents the model from collapsing into a 'predict-only-fake' bias.
    """
    def __init__(self, checkpoint_dir, video_ids, labels, batch_size, shuffle=True, augment=False):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        video_ids = np.array(video_ids)
        labels = np.array(labels)
        self.real_idx = np.where(labels == 0)[0]
        self.fake_idx = np.where(labels == 1)[0]
        
        self.video_ids = video_ids
        self.labels = labels

        n_real, n_fake = len(self.real_idx), len(self.fake_idx)
        print(f"   [generator] Real: {n_real}, Fake: {n_fake}")
        
        self.majority_count = max(n_real, n_fake)
        self.on_epoch_end()

    def __len__(self):
        # We define an epoch as seeing both classes equally based on the majority count
        return int(np.floor((2 * self.majority_count) / self.batch_size))

    def on_epoch_end(self):
        """Reshuffle and tile the minority class to match the majority."""
        if self.shuffle:
            np.random.shuffle(self.real_idx)
            np.random.shuffle(self.fake_idx)

        n_real, n_fake = len(self.real_idx), len(self.fake_idx)

        # Oversampling logic
        if n_real < n_fake:
            repeats = int(np.ceil(n_fake / n_real))
            epoch_real = np.tile(self.real_idx, repeats)[:n_fake]
            epoch_fake = self.fake_idx
        else:
            repeats = int(np.ceil(n_real / n_fake))
            epoch_fake = np.tile(self.fake_idx, repeats)[:n_real]
            epoch_real = self.real_idx

        # STRICT INTERLEAVING: [R, F, R, F, R, F...]
        # This is the "secret sauce" to stop the bias you saw in your evaluation.
        self.epoch_indices = []
        for r, f in zip(epoch_real, epoch_fake):
            self.epoch_indices.extend([r, f])

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_idx = self.epoch_indices[start:end]

        X, y = [], []
        for i in batch_idx:
            path = os.path.join(self.checkpoint_dir, f"{self.video_ids[i]}_{int(self.labels[i])}.npy")
            frames = np.load(path).astype(np.float32)

            if self.augment:
                # Basic flip augmentation
                if random.random() > 0.5:
                    frames = frames[:, :, ::-1, :].copy() 
                # Slight brightness jitter
                if random.random() > 0.5:
                    frames = np.clip(frames + random.uniform(-0.02, 0.02), 0, 1)

            X.append(frames)
            y.append(self.labels[i])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def make_sequence(samples, checkpoint_dir, batch_size, shuffle=True, augment=False):
    video_ids, labels = [], []
    for path, label in samples:
        vid = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(checkpoint_dir, f"{vid}_{label}.npy")):
            video_ids.append(vid)
            labels.append(label)
    return NpySequence(checkpoint_dir, video_ids, labels, batch_size, shuffle, augment)

# =============================================================================
# Training logic
# =============================================================================

def build_callbacks(tag, phase):
    ckpt_path = os.path.join(CONFIG["output_dir"], f"lightfakedetect_{tag}_phase{phase}_best.keras")
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["celeb", "dfdc"], default="celeb")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    apply_seeds()
    configure_cpu()
    make_dirs()
    
    # Load paths
    # Versioned checkpoints ensure we re-process when img_size or max_frames change
    ckpt_ver = f"checkpoints_s{CONFIG['img_size']}_f{CONFIG['max_frames']}"
    if args.dataset == "celeb":
        samples = load_video_paths(CONFIG["celeb_df_real_dir"], CONFIG["celeb_df_fake_dir"], max_videos=args.max_videos)
        ckpt_subdir = os.path.join(CONFIG["output_dir"], ckpt_ver, "celeb")
    else:
        samples = load_dfdc_paths(CONFIG["dfdc_dir"], max_videos=args.max_videos or 5000)
        ckpt_subdir = os.path.join(CONFIG["output_dir"], ckpt_ver, "dfdc")

    # Split
    labels = [s[1] for s in samples]
    train_val, test = train_test_split(samples, test_size=0.2, random_state=CONFIG["seed"], stratify=labels)
    train, val = train_test_split(train_val, test_size=0.2, random_state=CONFIG["seed"], stratify=[s[1] for s in train_val])

    # Preprocess check
    # We use save_only=True to prevent OOM on 16GB RAM; generator handles loading.
    os.makedirs(ckpt_subdir, exist_ok=True)
    build_dataset(train, "train", ckpt_subdir, is_train=False, save_only=True)
    build_dataset(val, "val", ckpt_subdir, is_train=False, save_only=True)
    build_dataset(test, "test", ckpt_subdir, is_train=False, save_only=True)

    # Build Model
    model = build_lightfakedetect(freeze_backbone=True)
    
    # Phase 1: Warm-up
    print("\n>>> Phase 1: Warm-up (MobileNet Frozen)")
    train_gen = make_sequence(train, ckpt_subdir, CONFIG["batch_size"], shuffle=True, augment=True)
    val_gen = make_sequence(val, ckpt_subdir, CONFIG["batch_size"], shuffle=False, augment=False)
    
    model.fit(train_gen, validation_data=val_gen, epochs=WARMUP_EPOCHS, callbacks=build_callbacks(args.dataset, 1))

    # Phase 2: Fine-tuning with LOWER learning rate (5e-6)
    print("\n>>> Phase 2: Fine-tuning (Unfrozen)")
    model = unfreeze_backbone(model, learning_rate=5e-6)
    
    remaining_epochs = CONFIG["epochs"] - WARMUP_EPOCHS
    history = model.fit(train_gen, validation_data=val_gen, epochs=remaining_epochs, callbacks=build_callbacks(args.dataset, 2))

    # Evaluation
    final_path = os.path.join(CONFIG["output_dir"], f"lightfakedetect_{args.dataset}.keras")
    model.save(final_path)
    
    print("\n[Final Evaluation]")
    X_test = np.array([np.load(os.path.join(ckpt_subdir, f"{os.path.splitext(os.path.basename(s[0]))[0]}_{s[1]}.npy")) for s in test], dtype=np.float32)
    y_test = np.array([s[1] for s in test], dtype=np.int32)
    evaluate_model(model, X_test, y_test, args.dataset)

if __name__ == "__main__":
    main()