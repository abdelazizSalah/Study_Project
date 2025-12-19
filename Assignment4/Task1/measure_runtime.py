import os
import threading
import time

import psutil
from keras.src.models import model

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



def measure_all(k):
    #RAW, RE15
    #measure RAM, runtime
    #feature extraction (autoencoder training)
    # training RAW (avg all folds):

    #avg_runtime, avg_peak_ram=train_and_save_models_rt("raw")
    #line=(
    #"=== Autoencoder Training Results (RAW feature type) ===\n"
    #"Averaged over all folds:\n"
    #f"  • Average runtime : {avg_runtime:.2f} s\n"
    #f"  • Peak RAM (max)  : {avg_peak_ram:.1f} MB\n"
    #"-----------------------------------------------\n\n"
    #)
    #with open("results/runtime_raw.txt", "a") as f:
    #    f.write(line)

    #feature extraction RAW (5 folds):
    avg_runtime, avg_peak_ram=create_features_for_ds_rt(k, "raw")

    line = (
        "=== Autoencoder Feature Extraction Results (RAW feature type) ===\n"
        "Averaged over all folds:\n"
        f"  • Average runtime : {avg_runtime:.2f} s\n"
        f"  • Peak RAM (max)  : {avg_peak_ram:.1f} MB\n"
        "-----------------------------------------------\n\n"
    )
    with open("results/runtime_raw.txt", "a") as f:
        f.write(line)

    #scenario 1
    #ee,lof,ocsvm
    #avg train, avg test

    #scenario 2
    #bsvm, rf, knn
    # avg train, avg test

    #scenario 3
    # bsvm, rf, knn
    # avg train, avg test

    # training (per fold)
    # testing (per fold)


    #training RE:
    #avg_runtime, avg_peak_ram=train_and_save_models_rt("re")

    #line = (
    #    "=== Autoencoder Training Results (RE15 feature type) ===\n"
    #    "Averaged over all folds:\n"
    #    f"  • Average runtime : {avg_runtime:.2f} s\n"
    #    f"  • Peak RAM (max)  : {avg_peak_ram:.1f} MB\n"
    #    "-----------------------------------------------\n\n"
    #)

    #with open("results/runtime_re15.txt", "a") as f:
    #    f.write(line)

    avg_runtime, avg_peak_ram=create_features_for_ds_rt(k, "re")

    line = (
        "=== Autoencoder Feature Extraction Results (RE15 feature type) ===\n"
        "Averaged over all folds:\n"
        f"  • Average runtime : {avg_runtime:.2f} s\n"
        f"  • Peak RAM (max)  : {avg_peak_ram:.1f} MB\n"
        "-----------------------------------------------\n\n"
    )

    with open("results/runtime_re15.txt", "a") as f:
        f.write(line)

    #training (per fold)
    #testing (per fold)
    return
