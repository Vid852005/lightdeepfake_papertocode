import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from config import CONFIG, apply_seeds, configure_cpu, make_dirs
from evaluate import evaluate_model, print_ablation_summary
from model import build_ablation_variant
from preprocess import build_dataset, load_video_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LightFakeDetect ablation study on Celeb-DF v2."
    )
    p.add_argument(
        "--max-videos", type=int, default=None,
        help="Cap number of videos (e.g. 200 for a 1-hour smoke-test)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_seeds()
    configure_cpu()
    make_dirs()

    print("\n" + "=" * 60)
    print("  LightFakeDetect — Ablation Study (Celeb-DF v2)")
    print("=" * 60 + "\n")
    samples = load_video_paths(
        real_dir=CONFIG["celeb_df_real_dir"],
        fake_dir=CONFIG["celeb_df_fake_dir"],
        max_videos=args.max_videos,
    )

    if not samples:
        print("[ERROR] No videos found — see data/ directory instructions.")
        return
    labels = [s[1] for s in samples]
    train_val, test = train_test_split(
        samples, test_size=0.20, random_state=CONFIG["seed"], stratify=labels
    )
    train, val = train_test_split(
        train_val, test_size=0.20, random_state=CONFIG["seed"],
        stratify=[s[1] for s in train_val],
    )

    ckpt_dir = os.path.join(CONFIG["output_dir"], "checkpoints", "ablation")
    X_train, y_train = build_dataset(train, "Ablation — train", ckpt_dir, is_train=True)
    X_val,   y_val   = build_dataset(val,   "Ablation — val  ", ckpt_dir, is_train=False)
    X_test,  y_test  = build_dataset(test,  "Ablation — test ", ckpt_dir, is_train=False)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    variants = ["full", "no_cbam", "no_gru", "standard_cnn"]
    ablation_results: dict = {}

    for variant in variants:
        print(f"\n{'─' * 50}")
        print(f"  Variant: {variant}")
        print("─" * 50)


        m = build_ablation_variant(
            variant=variant,
            seq_len=CONFIG["max_frames"],
            img_size=CONFIG["img_size"],
            gru_layers=CONFIG["gru_layers"],
            gru_units=CONFIG["gru_units"],
        )

        m.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            class_weight=class_weight_dict,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
            verbose=1,
        )

        metrics = evaluate_model(m, X_test, y_test, f"Ablation_{variant}")
        ablation_results[variant] = metrics

        save_path = os.path.join(CONFIG["output_dir"], f"ablation_{variant}.keras")
        m.save(save_path)
        print(f"  Model saved → {save_path}")
    print_ablation_summary(ablation_results)
    print(f"\nAblation complete. Outputs in: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
