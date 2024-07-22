#******************************************************************************************************************************************
#                                                                                                                                         *
#                           Deep Learning for Channel Coding via Neural Mutual Information Estimation                                     *
#                                                                                                                                         *
#    This code implements an **Autoencoder-based communication system** with mutual information estimation (MINE).                        *
#                                                                                                                                         *
#   1.  It simulates a communication channel where messages are encoded using a neural network-based encoder,                             *
#       transmitted through a noisy channel (AWGN or Rayleigh fading), and decoded by a neural network-based decoder.                     *
#                                                                                                                                         *
#   2.  The system utilizes a Mutual Information Neural Estimator (MINE) to optimize the encoder and estimate mutual information          *
#       between the transmitted and received signals. Performance evaluation includes calculating Batch Bit Error Rate (BER) and          *
#       comparing against theoretical limits (16-QAM simulation).                                                                         *
#                                                                                                                                         *
#   3.  This setup allows for studying and optimizing the communication system under different SNR conditions.                            *
#                                                                                                                                         *
#******************************************************************************************************************************************




import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from scipy import special
from tensorflow import keras
import matplotlib.pyplot as plt
assert sys.version_info >= (3, 5)


# --------------------------------------------------------------------- Configure matplotlib ---------------------------------------------------------------------------
# %matplotlib inline

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# -------------------------------------------------------------- Set random seeds for reproducibility ------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)
 

# -------------------------------------------- AutoEncoder class for constructing a communication system with an autoencoder -------------------------------------------

class AutoEncoder:
    def __init__(self, M=16, n=1, training_snr=7, rayleigh=False):
        self.M = M
        self.k = int(np.log2(M))
        self.n = n
        self.training_snr = training_snr
        self.rayleigh = rayleigh
        self.noise_std = self.EbNo_to_noise(training_snr)

        # Define custom layers
        self.norm_layer = keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))
        self.shape_layer = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2, n]))
        self.shape_layer2 = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2 * n]))
        self.channel_layer = keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std))

        # Define the encoder model
        self.encoder = keras.models.Sequential([
            keras.layers.Embedding(M, M, embeddings_initializer='glorot_normal', input_length=1),
            keras.layers.Dense(M, activation="elu"),
            keras.layers.Dense(2 * n, activation=None),
            self.shape_layer,
            self.norm_layer
        ])

        if rayleigh:                                                                                                        # Define the channel model
            self.channel = keras.models.Sequential([keras.layers.Lambda(lambda x: self.sample_Rayleigh_channel(x, self.noise_std))])
        else:
            self.channel = keras.models.Sequential([self.channel_layer])

        self.decoder = keras.models.Sequential([                                                                            # Define the decoder model
            keras.layers.InputLayer(input_shape=[2, n]),
            self.shape_layer2,
            keras.layers.Dense(M, activation="elu"),
            keras.layers.Dense(M, activation="softmax")
        ])

        self.autoencoder = keras.models.Sequential([self.encoder, self.channel, self.decoder])                              # Combine encoder, channel, and decoder into an autoencoder model
        