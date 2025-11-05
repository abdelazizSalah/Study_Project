from tensorflow import keras
import numpy as np

#print: model parameters
def extract_and_print_features_to_file(ds_all, file, model, model_config: str):
    encoder = keras.Model(model.input, model.get_layer("latent").output)    #todo: name for bottleneck layer has to be consisten!
    features = encoder.predict(ds_all, verbose=1)  # works for tf.data.Dataset or np arrays
    np.save(f"{file}_features.npy", features)
    #todo: log config into file next to features or at least the modl name


def plot_results(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"autoencoder_training.png")

    #plt.show()
    return


def evaluate(history, plot_results):
    if plot_results:
        plot_results(history)
    best_val_loss = min(history.history["val_loss"])
    best_train_loss = min(history.history["loss"])

    print(f"train_loss:{best_train_loss}; val_loss{best_val_loss} ")
    #todo: implement any other evaluation methods here

    return best_train_loss, best_val_loss