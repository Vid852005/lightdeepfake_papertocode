# =============================================================================
# predict.py — LightFakeDetect inference on one or more video files
#
# Extension: batch inference now reports total time and FPS, matching the
# format of paper Table 1 so results are directly comparable to the paper's
# A100 numbers (0.14–1.25 FPS on GPU).
#
# Usage
# -----
# Single video:
#   python predict.py --model outputs/lightfakedetect_celeb.keras \
#                     --video path/to/video.mp4
#
# Batch folder:
#   python predict.py --model outputs/lightfakedetect_celeb.keras \
#                     --folder path/to/videos/
# =============================================================================

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from config import CONFIG, configure_cpu
from model import CBAMLayer
from preprocess import preprocess_video


def load_model(model_path: str) -> tf.keras.Model:
    """Load a saved LightFakeDetect model (handles custom CBAMLayer)."""
    configure_cpu()
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"CBAMLayer": CBAMLayer}
    )
    return model


def predict_video(model: tf.keras.Model, video_path: str) -> dict:
    """Preprocess and predict a single video."""
    t0 = time.perf_counter()
    frames = preprocess_video(video_path, label=0)
    preprocess_time = time.perf_counter() - t0

    if frames is None:
        return {
            "path": video_path, "label": "ERROR",
            "confidence": None, "raw_score": None,
            "inference_ms": None,
        }

    X = frames[0]
    X = np.expand_dims(X, axis=0)

    t1 = time.perf_counter()
    score = float(model.predict(X, verbose=0)[0][0])
    inference_ms = (time.perf_counter() - t1) * 1000

    label      = "FAKE" if score >= 0.5 else "REAL"
    confidence = score if label == "FAKE" else 1.0 - score

    return {
        "path":           video_path,
        "label":          label,
        "confidence":     confidence,
        "raw_score":      score,
        "inference_ms":   inference_ms,
        "preprocess_ms":  preprocess_time * 1000,
    }


def predict_folder(model: tf.keras.Model, folder_path: str) -> list:
    """Run inference on all video files in a folder with FPS reporting."""
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not videos:
        print(f"No video files found in: {folder_path}")
        return []

    print(f"\n  {'File':<40}  {'Label':<6}  {'Confidence':>10}  {'Infer ms':>8}")
    print("  " + "─" * 72)

    t_wall = time.perf_counter()
    results = []

    for vp in videos:
        r = predict_video(model, vp)
        results.append(r)
        name = os.path.basename(vp)

        if r["label"] == "ERROR":
            print(f"  {name:<40}  {'ERROR':<6}  {'n/a':>10}  {'n/a':>8}")
        else:
            print(
                f"  {name:<40}  {r['label']:<6}  "
                f"{r['confidence']*100:>9.2f}%  "
                f"{r['inference_ms']:>7.1f}ms"
            )

    total_s   = time.perf_counter() - t_wall
    n_frames  = len(results) * CONFIG["max_frames"]
    fps       = n_frames / total_s if total_s > 0 else 0
    valid     = [r for r in results if r["label"] != "ERROR"]
    avg_infer = (
        sum(r["inference_ms"] for r in valid) / len(valid)
        if valid else 0
    )

    print(f"\n  {'─'*72}")
    print(f"  Processed : {len(results)} videos ({len(valid)} successful)")
    print(f"  Wall time : {total_s:.1f}s total")
    print(f"  Avg infer : {avg_infer:.1f} ms/video")
    print(f"  FPS       : {fps:.3f}  (paper A100 ref: 0.14–1.25 FPS)")
    print(f"  Note: CPU inference is expected to be slower than GPU.")

    return results


def print_result(r: dict) -> None:
    name = os.path.basename(r["path"])
    print("\n" + "=" * 56)
    print(f"  File        : {name}")
    if r["label"] == "ERROR":
        print("  Result      : ERROR — no face detected in video")
    else:
        print(f"  Prediction  : {r['label']}")
        print(f"  Confidence  : {r['confidence']*100:.2f}%")
        print(f"  Raw score   : {r['raw_score']:.6f}  (>= 0.5 → FAKE)")
        print(f"  Infer time  : {r['inference_ms']:.1f} ms")
        print(f"  Preprocess  : {r['preprocess_ms']:.1f} ms  (MTCNN face detection)")
    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description="LightFakeDetect inference")
    parser.add_argument("--model",  required=True, help="Path to .keras model file")
    parser.add_argument("--video",  default=None,  help="Path to a single video")
    parser.add_argument("--folder", default=None,  help="Path to a folder of videos")
    args = parser.parse_args()

    if args.video is None and args.folder is None:
        parser.error("Provide --video or --folder")

    model = load_model(args.model)

    if args.video:
        r = predict_video(model, args.video)
        print_result(r)

    if args.folder:
        print(f"\nBatch inference: {args.folder}")
        predict_folder(model, args.folder)


if __name__ == "__main__":
    main()
