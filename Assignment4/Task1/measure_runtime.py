import os
import threading
import time
from pathlib import Path

import psutil
from keras.src.models import model

from use_classifiers import execute_scenario_rt
from handling_re_bytes_integrated import get_keep_indices_from_fold0
from feature_creation_autoencoder import train_and_save_models_rt, create_features_for_ds_rt
from tensorflow import keras


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 ** 2)


def start_ram_monitor(interval=0.1):
    """
    Starts a background thread that tracks peak RSS of this process.
    Returns a handle (dict) that must be passed to stop_ram_monitor().
    """
    proc = psutil.Process(os.getpid())
    stop_event = threading.Event()
    peak_holder = {"peak": 0}

    def _monitor():
        while not stop_event.is_set():
            rss = proc.memory_info().rss
            if rss > peak_holder["peak"]:
                peak_holder["peak"] = rss
            time.sleep(interval)

        # final read
        rss = proc.memory_info().rss
        peak_holder["peak"] = max(peak_holder["peak"], rss)

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()

    return {
        "stop_event": stop_event,
        "thread": thread,
        "peak_holder": peak_holder,
    }


def stop_ram_monitor(handle):
    """
    Stops the RAM monitor and returns peak RSS in bytes.
    """
    handle["stop_event"].set()
    handle["thread"].join(timeout=2.0)
    return handle["peak_holder"]["peak"]



# ---------- small helpers (keep near measure_all) ----------

def _append(filepath: str, text: str) -> None:
    # creates file if missing; directory must exist
    with open(filepath, "a") as f:
        f.write(text)


def _block(title: str, feature_label: str, lines: list[str]) -> str:
    # nice, consistent formatting
    header = f"=== {title} ({feature_label}) ===\n"
    body = "\n".join(f"  • {ln}" for ln in lines) + "\n"
    return header + body + "-----------------------------------------------\n\n"


def _log_ae_training(feature_label: str, out_file: str, prefix_for_training: str) -> None:
    avg_runtime, avg_peak_ram = train_and_save_models_rt(prefix_for_training)

    _append(
        out_file,
        _block(
            "Autoencoder Training Results",
            feature_label,
            [
                "Averaged over all folds:",
                f"Average runtime : {avg_runtime:.2f} s",
                f"Peak RAM (max)  : {avg_peak_ram:.1f} MB",
            ],
        ),
    )


def _log_ae_feature_extraction(k: int, feature_label: str, out_file: str, prefix_for_features: str) -> None:
    avg_runtime, avg_peak_ram = create_features_for_ds_rt(k, prefix_for_features)

    _append(
        out_file,
        _block(
            "Autoencoder Feature Extraction Results",
            feature_label,
            [
                "Averaged over all folds:",
                f"Average runtime : {avg_runtime:.2f} s",
                f"Peak RAM (max)  : {avg_peak_ram:.1f} MB",
            ],
        ),
    )


def _log_classifiers_for_feature_type(
    k: int,
    global_label_encoder,
    prefix: str,                # "raw" or "re" (for execute_scenario_rt)
    feature_label: str,         # "RAW" or "RE15" (for printing)
    out_file: str,
    keep_indices: int = 0,
    param: int = 0,
):
    # scenario mapping
    scenario_models = {
        1: ["ocsvm", "lof", "ee"],
        2: ["rf", "knn", "bsvm"],
        3: ["rf", "knn", "bsvm"],
    }

    for scenario, models in scenario_models.items():
        for clf in models:
            rt_train, ram_train, rt_test, ram_test = execute_scenario_rt(
                global_label_encoder=global_label_encoder,
                classifier=clf,
                k=k,
                prefix=prefix,
                scenario=scenario,
                keep_indices=keep_indices,
                param=param,
            )

            _append(
                out_file,
                _block(
                    "Classifier Results",
                    feature_label,
                    [
                        f"Scenario: {scenario}",
                        f"Classifier: {clf}",
                        "Averaged over all folds:",
                        f"Train avg runtime : {rt_train:.2f} s",
                        f"Train peak RAM    : {ram_train:.1f} MB",
                        f"Test  avg runtime : {rt_test:.2f} s",
                        f"Test  peak RAM    : {ram_test:.1f} MB",
                    ],
                ),
            )


# ---------- main function ----------

def measure_all(k: int, global_label_encoder) -> None:
    """
    Runs everything once for RAW and once for RE15, measures runtime and peak RAM
      - AE training
      - AE feature extraction
      - Scenario 1: ocsvm, lof, ee
      - Scenario 2 & 3: rf, knn, bsvm
    Logs results to:
      - results/runtime_raw.txt
      - results/runtime_re15.txt
    """
    Path("results").mkdir(exist_ok=True)

    # ---------------- RAW ----------------
    raw_file = "results/runtime_raw.txt"
    _log_ae_training(feature_label="RAW feature type", out_file=raw_file, prefix_for_training="raw")
    _log_ae_feature_extraction(k=k, feature_label="RAW feature type", out_file=raw_file, prefix_for_features="raw")

    _log_classifiers_for_feature_type(
        k=k,
        global_label_encoder=global_label_encoder,
        prefix="raw",
        feature_label="RAW feature type",
        out_file=raw_file,
        keep_indices=0,
        param=0,
    )

    # ---------------- RE15 ----------------
    re_file = "results/runtime_re15.txt"

    # AE training for RE (your train_and_save_models_rt uses prefix "re" to load the right base model)
    _log_ae_training(feature_label="RE15 feature type", out_file=re_file, prefix_for_training="re")
    _log_ae_feature_extraction(k=k, feature_label="RE15 feature type", out_file=re_file, prefix_for_features="re")

    keep_indices = get_keep_indices_from_fold0("datasets/re_bytes_15/re_features_fold0", "re15")

    _log_classifiers_for_feature_type(
        k=k,
        global_label_encoder=global_label_encoder,
        prefix="re",  # IMPORTANT: was wrongly "raw" in your old code
        feature_label="RE15 feature type",
        out_file=re_file,
        keep_indices=keep_indices,
        param=15,
    )
