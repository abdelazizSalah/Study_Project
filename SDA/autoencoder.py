from tensorflow import keras
from tensorflow.keras import layers


def build_tiny_ae(input_dim: int, bottleneck: int = 32, activation: str = "relu", noise_std: float = 0.02) -> keras.Model:
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