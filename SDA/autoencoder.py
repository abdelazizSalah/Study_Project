from tensorflow import keras
from tensorflow.keras import layers
from evaluation_results import evaluate


#dense=fully connected
def build_tiny_ae_dense(input_dim: int, bottleneck: int = 32, activation: str = "relu", noise_std: float = 0.02) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="bytes")
    x = layers.GaussianNoise(noise_std, name="noise")(inp)
    # Encoder
    x = layers.Dense(128, activation=activation, name="enc1")(x)
    z = layers.Dense(bottleneck, activation=None, name="latent")(x)
    # Decoder
    x = layers.Dense(128, activation=activation, name="dec1")(z)
    out = layers.Dense(input_dim, activation="sigmoid", name="recon")(x)
    model = keras.Model(inp, out, name="TinyAE")
    return model


#conv1d=?
def build_tiny_ae_conv1d(input_dim: int, bottleneck: int = 32, activation: str = "relu", noise_std: float = 0.02):
    #todo
    pass


def create_train_evaluate_model(ds_train, ds_test, activation: str, layer_type: str, param_settings):
    # learning rate = size of single weight update step
    # higher lr -> faster training, lr too high -> training unstable

    if layer_type=="dense":
        model = build_tiny_ae_dense(activation=activation)
    else:
        model = build_tiny_ae_conv1d(activation=activation)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    # define early stopping callback
    # if loss doesn't improve for 5 consecutive epochs -> roll back to weights with lowest loss
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    # training
    history = model.fit(ds_train, validation_data=ds_test, epochs=100, callbacks=cb, verbose=1)


    return model, min(history.history["val_loss"]) #minimum MSE for test data of that model


def hyperparameter_search(ds_train, ds_test, activation: str, layer_type: str):
    #todo: implement grid search of optimal hyperparameter for each model

    best_val_loss=1000
    best_model=None
    best_param_settings=None

    #todo: loop - foreach hyper parameter setting do:
    param_settings=[] #todo
    model, val_loss= create_train_evaluate_model(ds_train, ds_test, activation, layer_type, param_settings)


    if val_loss < best_val_loss:
        best_model=model
        best_param_settings=param_settings

    #print features of best model to file;


    return best_model, str(best_param_settings) #return model with best hyper parameter settings (for printing)
