import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CONFIG


def evaluate_model(
    model,
    X_test:       np.ndarray,
    y_test:       np.ndarray,
    dataset_name: str = "Dataset",
) -> dict:
    n_timing = min(50, len(X_test))
    times: list[float] = []
    for i in range(n_timing):
        t0 = time.perf_counter()
        model.predict(X_test[i:i+1], verbose=0)
        times.append(time.perf_counter() - t0)
    stable_times = times[5:] if len(times) > 5 else times
    avg_ms       = np.mean(stable_times) * 1000
    fps          = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    y_prob = model.predict(X_test, batch_size=CONFIG["batch_size"]).flatten().astype("float32")
    y_pred = (y_prob >= 0.5).astype(int)

    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)
    real_idx = np.where(y_test == 0)[0]
    fake_idx = np.where(y_test == 1)[0]
    real_correct = np.sum(y_pred[real_idx] == 0)
    fake_correct = np.sum(y_pred[fake_idx] == 1)
    real_as_fake = len(real_idx) - real_correct
    
    fake_as_real = len(fake_idx) - fake_correct
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Evaluation — {dataset_name}")
    print(sep)
    print(f"  Recall            : {rec     * 100:.2f}%")
    print(f"  Precision         : {prec    * 100:.2f}%")
    print(f"  F1-Score          : {f1      * 100:.2f}%")
    print(f"  Accuracy          : {acc     * 100:.2f}%")
    print(f"  Balanced Accuracy : {bal_acc * 100:.2f}%")
    print(f"  AUC               : {auc     * 100:.2f}%")
    print(f"  Inference Time    : {avg_ms:.1f} ms/frame  ({fps:.2f} FPS)")
    print(f"  [Paper A100 ref]  : 3–7 s/frame  (0.14–0.33 FPS)")
    print(f"\n  Per-class breakdown:")
    print(f"    Real correctly classified : {real_correct}/{len(real_idx)}")
    print(f"    Real misclassified as fake: {real_as_fake}/{len(real_idx)}"
          f"  ← paper notes this is the dominant error")
    print(f"    Fake correctly classified : {fake_correct}/{len(fake_idx)}")
    print(f"    Fake misclassified as real: {fake_as_real}/{len(fake_idx)}")
    print(f"{sep}\n")
    tag = dataset_name.replace(" ", "_")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluation — {dataset_name}", fontsize=14)

    cm = confusion_matrix(y_test, y_pred, normalize="true") * 100
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    axes[0].set_title("Normalised Confusion Matrix (%)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(CONFIG["output_dir"], f"eval_{tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")

    return dict(
        acc=acc, bal_acc=bal_acc, prec=prec, rec=rec,
        f1=f1, auc=auc, fps=fps, ms_per_frame=avg_ms,
    )


def plot_training_curves(history, dataset_name: str = "Dataset") -> None:
    tag = dataset_name.replace(" ", "_")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {dataset_name}", fontsize=14)

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", linestyle="--")
    axes[1].set_title("Accuracy")
    
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    out_path = os.path.join(CONFIG["output_dir"], f"training_curves_{tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved → {out_path}")


def print_ablation_summary(ablation_results: dict) -> None:
    print("\nAblation Study Summary")
    print(f"{'Variant':<20} {'Recall':>8} {'Precision':>10} "
          f"{'Accuracy':>10} {'AUC':>8} {'FPS':>7}")
    print("-" * 68)
    for variant, r in ablation_results.items():
        fps_str = f"{r.get('fps', 0):.2f}" if r.get('fps') else "  n/a"
        print(
            f"{variant:<20} "
            f"{r['rec']  * 100:>7.2f}%  "
            f"{r['prec'] * 100:>9.2f}%  "
            f"{r['acc']  * 100:>9.2f}%  "
            f"{r['auc']  * 100:>7.2f}%  "
            f"{fps_str:>7}"
        )
