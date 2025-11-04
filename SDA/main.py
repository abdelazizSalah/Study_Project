#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

from Anna_Code.file_helper import list_files_by_filetype
from SDA.autoencoder import build_tiny_ae
from SDA.prepare_data import create_matrix_from_pcaps, make_datasets

from tensorflow import keras
from tensorflow.keras import layers


def evaluate():
    pass


def plot_results(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"autoencoder_training.png")

    #plt.show()

#/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set
def main():
    p = argparse.ArgumentParser(description="Minimal Autoencoder for ICS packets")
    p.add_argument("--pcap-dir", type=Path, help="Directory with PCAP/PCAPNG files (recursive)")
    p.add_argument("--M", type=int, required=True, help="Bytes per packet (input length)")
    p.add_argument("--epochs", required=True, type=int, help="")

    args = p.parse_args()

    packet_length_M = args.M
    files=list_files_by_filetype(args.pcap_dir, "pcap")
    matrix=create_matrix_from_pcaps(files, packet_length_M)
    print(matrix.shape)

    #todo batch siz and train ratio as parameters
    ds_train, ds_test, X_train, X_test = make_datasets(matrix)

    #create model
    model = build_tiny_ae(input_dim=packet_length_M)
    #learning rate = size of single weight update step
    #higher lr -> faster training, lr too high -> training unstable
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    print(model.summary())

    #define early stopping callback
    #if loss doesn't improve for 5 consecutive epochs -> roll back to weights with lowest loss
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    # train model
    history = model.fit(ds_train, validation_data=ds_test, epochs=args.epochs, callbacks=cb, verbose=1)
    plot_results(history)

    best_val_loss = min(history.history["val_loss"])
    best_train_loss = min(history.history["loss"])
    print(f"train_loss:{best_train_loss}; val_loss{best_val_loss} ")

if __name__ == "__main__":
    main()
